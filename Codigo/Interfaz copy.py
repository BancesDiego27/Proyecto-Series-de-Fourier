# app.py
"""
Serie de Fourier Truncada — v4 (entrada por tramos con a, b numéricos)
Autor: Diego Bances Mejía (adaptado por asistente)
Descripción:
- En lugar de parsear condiciones en texto, el usuario ingresa a y b para cada tramo (numéricos)
  y la expresión f(t) en ese tramo.
- Construcción directa de la función por tramos y cálculo de coeficientes con numpy.trapezoid.
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from PIL import Image
import cv2

tab1, tab2 = st.tabs(["Parte I — Fourier Truncada", "Parte II — FFT en Imágenes"])
with tab1:
    st.set_page_config(page_title="Serie de Fourier Truncada — Parte I (v4)", layout="wide")

    # ---------------------------
    # Helpers
    # ---------------------------
    SAFE_MATH_NAMES = {
        'np': np,
        'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
        'exp': np.exp, 'sqrt': np.sqrt, 'abs': np.abs,
        'pi': np.pi, 'e': np.e,
        'arcsin': np.arcsin, 'arccos': np.arccos, 'arctan': np.arctan,
        'sinh': np.sinh, 'cosh': np.cosh, 'tanh': np.tanh,
        'sign': np.sign
    }
    def eval_expr(expr: str, t_array: np.ndarray):
        """
        Evalúa la expresión expr en el arreglo t_array y devuelve un array de floats.
        Permite usar 'np' o nombres como sin, cos (mapeados a numpy).
        Si el resultado es un escalar, se convierte a un array constante.
        """
        local_env = {**SAFE_MATH_NAMES, 't': t_array}
        try:
            result = eval(expr, {"__builtins__": {}}, local_env)
            if np.isscalar(result):
                result = np.full_like(t_array, float(result), dtype=float)
            else:
                result = np.array(result, dtype=float)
            return result
        except Exception as e:
            st.warning(f"Error evaluando '{expr}': {e}")
            return np.zeros_like(t_array, dtype=float)


    def build_piecewise_from_numeric(tramos):
        """
        tramos: lista de dicts {'a': float, 'b': float, 'expr': str}
        Retorna:
        - func(t_array) -> y_array
        - a_min, b_max (floats)
        - mensajes de validacion (lista)
        Reglas:
        - Para cada tramo se asigna donde (t >= a) & (t < b).
        - Para el tramo con el máximo b, se hace (t >= a) & (t <= b) para incluir el extremo derecho.
        """
        msgs = []
        # Validar y ordenar tramos por a
        tramos_sorted = sorted(tramos, key=lambda x: x['a'])
        # Determinar a_min y b_max
        a_vals = [tr['a'] for tr in tramos_sorted]
        b_vals = [tr['b'] for tr in tramos_sorted]
        a_min = min(a_vals) if a_vals else -np.pi
        b_max = max(b_vals) if b_vals else np.pi
        if any(tr['b'] <= tr['a'] for tr in tramos_sorted):
            msgs.append("Atención: hay tramos con b <= a; revísalos.")
        # Check for overlaps/gaps (informativo)
        for i in range(len(tramos_sorted)-1):
            if tramos_sorted[i]['b'] < tramos_sorted[i+1]['a']:
                msgs.append(f"Hueco detectado entre tramo {i+1} y {i+2}: [{tramos_sorted[i]['b']}, {tramos_sorted[i+1]['a']}]")
            if tramos_sorted[i]['b'] > tramos_sorted[i+1]['a']:
                msgs.append(f"Solapamiento detectado entre tramo {i+1} y {i+2}.")

        def func(t_input):
            t = np.array(t_input, dtype=float)
            y = np.zeros_like(t, dtype=float)
            # last b to include right endpoint
            b_last = tramos_sorted[-1]['b'] if tramos_sorted else b_max
            for idx, tr in enumerate(tramos_sorted):
                a_i = float(tr['a'])
                b_i = float(tr['b'])
                expr = tr['expr']
                if idx == len(tramos_sorted)-1:
                    mask = (t >= a_i) & (t <= b_i)
                else:
                    mask = (t >= a_i) & (t < b_i)
                if np.any(mask):
                    try:
                        vals = eval_expr(expr, t)  # eval full vector
                        y[mask] = np.array(vals, dtype=float)[mask]
                    except Exception as e:
                        # si hay error en la evaluación, marcar y dejar ceros en ese tramo
                        msgs.append(f"Error evaluando expr en tramo {idx+1} ('{expr}'): {e}")
                # si no hay puntos en la máscara, no hacemos nada
            return y

        return func, a_min, b_max, msgs

    def compute_coeffs(func, a, b, N, M=4000):
        """Calcula a0, an, bn, Ef usando numpy.trapezoid"""
        T = b - a
        omega = 2*np.pi / T
        t = np.linspace(a, b, M, endpoint=False)
        y = func(t)
        Ef = np.trapezoid(y**2, t)
        a0 = (1.0/T) * np.trapezoid(y, t)
        an = np.zeros(N)
        bn = np.zeros(N)
        for n in range(1, N+1):
            an[n-1] = (2.0/T) * np.trapezoid(y * np.cos(n*omega*t), t)
            bn[n-1] = (2.0/T) * np.trapezoid(y * np.sin(n*omega*t), t)
        return a0, an, bn, Ef, t, y

    def fourier_eval(a0, an, bn, T, t_eval):
        omega = 2*np.pi / T
        y = np.full_like(t_eval, a0, dtype=float)
        for n, (a_n, b_n) in enumerate(zip(an, bn), start=1):
            y += a_n * np.cos(n*omega*t_eval) + b_n * np.sin(n*omega*t_eval)
        return y

    def coeffs_df(a0, an, bn):
        df = pd.DataFrame({
            'n': [0] + list(range(1, len(an)+1)),
            'a_n': [a0] + list(an),
            'b_n': [0.0] + list(bn)
        })
        return df

    def csv_link(df, filename="coeficientes_fourier.csv"):
        buf = BytesIO()
        df.to_csv(buf, index=False)
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode()
        return f"data:file/csv;base64,{b64}"

    # ---------------------------
    # UI (única vista)
    # ---------------------------
    st.title("Serie de Fourier Truncada — Parte I (v4)")
    st.markdown("Define la función por tramos usando valores numéricos para cada tramo: inicio `a`, fin `b` y la expresión en `t` (ej: `0`, `t+2`, `np.sin(t)`, `sin(t)`). Se detecta automáticamente el intervalo total y el período.")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.header("Definir función por tramos")
        modo = st.selectbox("Modo", ["f1 predefinida (rectangular)", "f2 predefinida (rampa)", "Personalizada (numérica)"])
        tramos = []
        if modo.startswith("f1"):
            st.latex(r"f_1(t)=\begin{cases}0 & -2<t<-1 \\ 1 & -1<t<1 \\ 0 & 1<t<2\end{cases}")
            tramos = [
                {'a': -2.0, 'b': -1.0, 'expr': '0'},
                {'a': -1.0, 'b':  1.0, 'expr': '1'},
                {'a':  1.0, 'b':  2.0, 'expr': '0'}
            ]
        elif modo.startswith("f2"):
            st.latex(r"f_2(t)=\begin{cases}0 & -2<t<-1 \\ t+2 & -1<t<1 \\ 0 & 1<t<2\end{cases}")
            tramos = [
                {'a': -2.0, 'b': -1.0, 'expr': '0'},
                {'a': -1.0, 'b':  1.0, 'expr': 't + 2'},
                {'a':  1.0, 'b':  2.0, 'expr': '0'}
            ]
        else:
            st.markdown("Ingresa el número de tramos y luego completa `a`, `b` y la expresión en t para cada tramo.")
            n_tramos = st.number_input("Número de tramos:", min_value=1, max_value=12, value=3, step=1)
            for i in range(int(n_tramos)):
                st.markdown(f"**Tramo {i+1}**")
                a_i = st.number_input(f"a_{i+1} (inicio):", value=-2.0 + i*1.0, key=f"a_{i}")
                b_i = st.number_input(f"b_{i+1} (fin):", value=-1.0 + i*1.0, key=f"b_{i}")
                expr_i = st.text_input(f"Expr tramo {i+1} (ej: 0, t+2, np.sin(t)):", value=("0" if i!=1 else "t + 2"), key=f"expr_{i}")
                tramos.append({'a': a_i, 'b': b_i, 'expr': expr_i})

        st.markdown("---")
        st.header("Parámetros de cálculo")
        N = st.slider("Número de armónicos N:", min_value=1, max_value=120, value=10)
        M = st.number_input("Puntos para integración (M):", min_value=200, max_value=200000, value=4000, step=200)
        st.markdown("Botones rápidos:")
        c1, c2, c3 = st.columns(3)
        if c1.button("N = 3"):
            N = 3
        if c2.button("N = 5"):
            N = 5
        if c3.button("N = 10"):
            N = 10

        st.markdown("---")
        st.markdown("**Información y ayuda**")
        st.markdown(r"""
        - Los coeficientes se calculan numéricamente usando la regla del trapecio sobre una malla densa.  
        - Fórmulas usadas:  
        - $a_0 = \dfrac{1}{T}\int_a^{a+T} f(t)\,dt$  
        - $a_n = \dfrac{2}{T}\int_a^{a+T} f(t)\cos(n\omega t)\,dt$  
        - $b_n = \dfrac{2}{T}\int_a^{a+T} f(t)\sin(n\omega t)\,dt$ con $\omega=\dfrac{2\pi}{T}$  
        - ICE(N) según el enunciado:  
        
        $$\text{ICE}(N) = E_f - \left( a_0^2 T + \frac{T}{2} \sum_{n=1}^{N} (a_n^2 + b_n^2) \right)$$
        """)
        st.markdown("Se usa `numpy.trapezoid` para las integrales (eficiente para mallas densas).")

    with col2:
        st.header("Resultados y visualización")

        # Construir función por tramos (numéricos)
        func, a_tot, b_tot, messages = build_piecewise_from_numeric(tramos)

        # Previsualización de la función definida (un periodo)
        st.subheader("Previsualización de la función (un periodo)")
        t_preview = np.linspace(a_tot, b_tot, 2000, endpoint=False)
        y_preview = func(t_preview)

        fig_p, ax_p = plt.subplots(figsize=(9,3))
        ax_p.plot(t_preview, y_preview, linewidth=1.5)
        ax_p.set_xlabel("t")
        ax_p.grid(True)
        ax_p.set_title("Función definida por tramos")
        st.pyplot(fig_p)


        T = b_tot - a_tot
        if T <= 0:
            st.error("El intervalo total detectado no es válido (T <= 0). Revisa los valores a/b de los tramos.")
            st.stop()

        # Mostrar mensajes de validación si existen
        if messages:
            st.subheader("Advertencias / mensajes")
            for m in messages:
                st.warning(m)



        # Cálculo de coeficientes
        st.subheader("Cálculo de coeficientes")
        with st.spinner("Calculando..."):
            try:
                a0, an, bn, Ef, t_malla, y_true = compute_coeffs(func, a_tot, b_tot, N, M=M)
            except Exception as e:
                st.error(f"Error en el cálculo numérico: {e}")
                st.stop()

        series_energy = a0**2 * T + (T/2.0) * np.sum(an**2 + bn**2)
        ICE = Ef - series_energy
        ICE_ratio = ICE / Ef if Ef != 0 else np.nan
        omega = 2*np.pi / T
        colA, colB = st.columns(2)
        with colA:
            st.write(f"Intervalo detectado: [{a_tot}, {b_tot}]")
            st.write(f"Período T = {T:.6g}")
            st.write(f"Omega ω = 2π/T = {omega:.6g}")
            st.write(f"Energía Ef ≈ {Ef:.6e}")
            st.write(f"a₀ ≈ {a0:.6e}")
        with colB:
            st.write(f"Energía por la serie (hasta N): {series_energy:.6e}")
            st.write(f"ICE(N) = Ef - [...] ≈ {ICE:.6e}")
            st.write(f"ICE / Ef = {ICE_ratio:.2%}")
            if ICE <= 0.02 * Ef:
                st.success("ICE ≤ 0.02 Ef (condición satisfecha)")
            else:
                st.warning("ICE > 0.02 Ef")

        st.markdown("---")
        st.subheader("Ecuación truncada (con ω y primeros términos)")
        st.latex(r"f_N(t) = a_0 + \sum_{n=1}^{N} \left[ a_n \cos(n\omega t) + b_n \sin(n\omega t)\right], \quad \omega = \frac{2\pi}{T}")
        # Mostrar un ejemplo con valores numéricos para los primeros términos
        def build_latex_values(a0, an, bn, show=6):
            s = f"{a0:.4e} "
            upto = min(len(an), show)
            for n in range(1, upto+1):
                s += f"+ ({an[n-1]:.4e})\\cos({n}\\omega t) + ({bn[n-1]:.4e})\\sin({n}\\omega t) "
            if len(an) > show:
                s += "+ \\dots"
            return s
        st.latex(build_latex_values(a0, an, bn, show=10))

        # Gráfica: original extendida y serie truncada (varios periodos)
        st.subheader("Original (extendida) vs Serie truncada")
        t_plot = np.linspace(a_tot - 0.5*T, b_tot + 0.5*T, 3000)
        # repetir la señal original periódicamente usando modulo
        t_mapped = ((t_plot - a_tot) % T) + a_tot
        y_periodic = func(t_mapped)
        y_series = fourier_eval(a0, an, bn, T, t_plot)

        fig_s, ax_s = plt.subplots(figsize=(10,3.5))
        ax_s.plot(t_plot, y_periodic, label="Original (período repetido)", linewidth=1.5)
        ax_s.plot(t_plot, y_series, linestyle='--', label=f"Serie truncada (N={N})", linewidth=1.2)
        ax_s.set_xlabel("t")
        ax_s.legend()
        ax_s.grid(True)
        st.pyplot(fig_s)

        # Residuo en un periodo
        st.subheader("Residuo en un período (f - f_N)")
        t_res = np.linspace(a_tot, b_tot, 2000, endpoint=False)
        resid = func(t_res) - fourier_eval(a0, an, bn, T, t_res)
        fig_r, ax_r = plt.subplots(figsize=(9,2.5))
        ax_r.plot(t_res, resid)
        ax_r.grid(True)
        ax_r.set_title("Residuo f(t) - f_N(t)")
        st.pyplot(fig_r)

        # Tabla de coeficientes y descarga CSV
        st.subheader("Coeficientes numéricos")
        df_coef = coeffs_df(a0, an, bn)
        st.dataframe(df_coef, use_container_width=True)
        href = csv_link(df_coef)
        st.markdown(f"[Descargar coeficientes (CSV)]({href})")

        # Evolución ICE con N de ejemplo
        st.subheader("Evolución de ICE(N) — muestras")
        Ns_test = [1,2,3,5,10,20,40,80]
        Ns_test = [n for n in Ns_test if n <= 200]
        ICE_vals = []
        for ntest in Ns_test:
            a0t, ant, bnt, Eft, _, _ = compute_coeffs(func, a_tot, b_tot, ntest, M=M)
            series_energy_t = a0t**2 * T + (T/2.0) * np.sum(ant**2 + bnt**2)
            ICE_vals.append(Eft - series_energy_t)

        fig_ice, ax_ice = plt.subplots(figsize=(8,3))
        ax_ice.plot(Ns_test, ICE_vals, marker='o')
        ax_ice.axhline(0.02 * Ef, color='red', linestyle='--', label='0.02 * Ef')
        ax_ice.set_xlabel("N")
        ax_ice.set_ylabel("ICE(N)")
        ax_ice.grid(True)
        ax_ice.legend()
        st.pyplot(fig_ice)

    st.markdown("---")
    st.caption("Integrantes: ")

with tab2:
    def detectar_blur_fft(img_gray, size=60):
        f = np.fft.fft2(img_gray)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = np.abs(fshift)

        crow, ccol = img_gray.shape[0]//2, img_gray.shape[1]//2
        magnitude_spectrum[crow-size:crow+size, ccol-size:ccol+size] = 0

        # Energía total vs energía alta frecuencia
        total_energy = np.sum(np.abs(fshift))
        high_freq_energy = np.sum(magnitude_spectrum) / total_energy

        return high_freq_energy



    st.title("Parte II – Filtros en Imágenes (FFT)")

    uploaded = st.file_uploader("Sube una imagen", type=["jpg", "png", "jpeg"])

    filtro_tipo = st.selectbox("Tipo de filtro", ["Pasa bajas", "Pasa altas"])
    frecuencia_corte = st.slider("Frecuencia de corte (f_cutoff)", 0.0, 1.0, 0.3, 0.01)

    if uploaded:
        # --- Leer imagen y convertir a float
        img = cv2.imdecode(np.frombuffer(uploaded.read(), np.uint8), cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        blur_score = detectar_blur_fft(img_gray)

        if blur_score < 0.5:  
            st.warning(f"La imagen parece borrosa (score={blur_score:.2f})")
        else:
            st.success(f"La imagen está nítida (score={blur_score:.2f})")

        # --- FFT por canal (R, G, B)
        canales_filtrados = []
        espectro_magnitud = None
        rows, cols, _ = img_rgb.shape
        crow, ccol = rows // 2, cols // 2
        mask = np.zeros((rows, cols, 3), np.float32)

        # --- Crear la máscara (pasa bajas / pasa altas)
        radius = int(frecuencia_corte * min(crow, ccol))
        Y, X = np.ogrid[:rows, :cols]
        dist = np.sqrt((X - ccol)**2 + (Y - crow)**2)

        if filtro_tipo == "Pasa bajas":
            mask = np.repeat((dist <= radius).astype(np.float32)[:, :, np.newaxis], 3, axis=2) 

        else:
            mask = np.repeat((dist > radius).astype(np.float32)[:, :, np.newaxis], 3, axis=2)


        #  Aplicar FFT a cada canal
        for i in range(3):
            F = np.fft.fft2(img_rgb[:, :, i]) #Transformada a dominio frecuencia "2D"
            F_shift = np.fft.fftshift(F)  #centra la frecuencia cero
            F_filtered = F_shift * mask[:, :, i] #aplica la máscara de filtro
            F_ishift = np.fft.ifftshift(F_filtered) #descentra la frecuencia cero
            img_back = np.abs(np.fft.ifft2(F_ishift)) #Transformada inversa a dominio espacial
            img_back = np.clip(img_back / np.max(img_back), 0, 1) #normalizado
            canales_filtrados.append(img_back) #Guarda canal filtrado
        
            if i == 0:  # mostrar espectro solo una vez
                espectro_magnitud = np.log(1 + np.abs(F_shift)) #Espectro de magnitud
                espectro_magnitud = espectro_magnitud / np.max(espectro_magnitud) #normalizado

        # Combinar canales para imagen final
        img_filtered = np.dstack(canales_filtrados)
        img_filtered = np.clip(img_filtered, 0, 1)

        col1, col2, col3 = st.columns(3)
        col1.image(img_rgb, caption="Original", use_container_width=True)
        col2.image(espectro_magnitud, caption="Espectro", use_container_width=True, clamp=True)
        col3.image(img_filtered, caption=f"Filtrada ({filtro_tipo}, f_c={frecuencia_corte})", use_container_width=True, clamp=True)

        
        st.markdown("### Analisis del filtro aplicado")
        if filtro_tipo == "Pasa bajas":
            st.info("El filtro pasa bajas elimina las altas frecuencias → la imagen se suaviza y pierde detalles finos.")
        else:
            st.info("El filtro pasa altas elimina las bajas frecuencias → la imagen resalta bordes y detalles, pero pierde regiones suaves.")