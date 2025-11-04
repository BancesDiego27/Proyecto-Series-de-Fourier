import streamlit as st
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from PIL import Image
from numpy.fft import fft2, fftshift, ifft2
from sympy import Piecewise
from scipy import integrate
import matplotlib.pyplot as plt
import cv2
print(cv2.__version__)

st.title("Análisis de Fourier: Series y FFT")

section = st.sidebar.radio("Selecciona la sección:", 
                           ["Parte I - Series de Fourier", 
                            "Parte II - FFT e Imágenes"])

if section == "Parte I - Series de Fourier":
    
    st.set_page_config(page_title="Fourier Series - Función por tramos", layout="wide")
    st.title("Series de Fourier: Funciones por tramos (2N + 1 términos)")

    st.markdown("""
    Esta app calcula la **serie trigonométrica de Fourier** de una función **por tramos** definida en un intervalo.
    Se muestran los coeficientes, la serie truncada, el período y la energía (por Parseval).

    **Instrucciones:**
    1. Ingrese cuántos tramos tiene la función.
    2. Especifique para cada tramo su expresión y el intervalo donde aplica.
    3. Ajuste el número de armónicos (N) y presione **Calcular**.
    """)

    # --- Sidebar Inputs ---
    st.sidebar.header("Entrada de datos")

    num_tramos = st.sidebar.number_input("Número de tramos", min_value=1, max_value=10, value=2, step=1)

    # Guardamos los tramos en una lista
    tramos = []
    for i in range(num_tramos):
        st.sidebar.markdown(f"**Tramo {i+1}:**")
        expr_text = st.sidebar.text_input(f"f{i+1}(x) =", "1" if i == 0 else "-1", key=f"expr_{i}")
        a_i = st.sidebar.text_input(f"Inicio (a{i+1})", "-pi" if i == 0 else "0", key=f"a_{i}")
        b_i = st.sidebar.text_input(f"Fin (b{i+1})", "0" if i == 0 else "pi", key=f"b_{i}")
        tramos.append((expr_text, a_i, b_i))

    N = st.sidebar.slider("N (armónicos)", min_value=0, max_value=50, value=5)
    resolution = st.sidebar.slider("Puntos para graficar", 200, 5000, 1000)
    compute_btn = st.sidebar.button("Calcular")

    x = sp.symbols('x')

    # --- Funciones auxiliares ---
    def construir_piecewise(tramos):
        """
        Construye una expresión SymPy Piecewise a partir de los tramos ingresados por el usuario.
        """
        condiciones = []
        for expr_text, a_i, b_i in tramos:
            try:
                expr = sp.sympify(expr_text)
                a_val = sp.sympify(a_i)
                b_val = sp.sympify(b_i)
                cond = (x >= a_val) & (x < b_val)
                condiciones.append((expr, cond))
            except Exception as e:
                st.error(f"Error al procesar el tramo: {expr_text} en [{a_i}, {b_i}] — {e}")
                return None
        return Piecewise(*condiciones)

    def integrar(func, a, b):
        result, _ = integrate.quad(lambda t: float(func(t)), a, b, limit=200)
        return result

    def compute_coefficients(f_np, a, b, N):
        T = b - a
        A0 = (1.0 / T) * integrar(f_np, a, b)
        An, Bn = [], []
        for n in range(1, N + 1):
            cos_fun = lambda t, n=n: f_np(t) * np.cos(2 * np.pi * n * (t - a) / T)
            sin_fun = lambda t, n=n: f_np(t) * np.sin(2 * np.pi * n * (t - a) / T)
            an = (2.0 / T) * integrate.quad(lambda t: float(cos_fun(t)), a, b, limit=200)[0]
            bn = (2.0 / T) * integrate.quad(lambda t: float(sin_fun(t)), a, b, limit=200)[0]
            An.append(an)
            Bn.append(bn)
        return A0, np.array(An), np.array(Bn)

    def truncated_series_eval(A0, An, Bn, a, b, xs):
        T = b - a
        y = np.full_like(xs, A0, dtype=float)
        for idx, n in enumerate(range(1, len(An) + 1)):
            y += An[idx] * np.cos(2 * np.pi * n * (xs - a) / T) + Bn[idx] * np.sin(2 * np.pi * n * (xs - a) / T)
        return y

    # --- Main computation ---
    if compute_btn:
        pw_expr = construir_piecewise(tramos)
        if pw_expr is None:
            st.stop()
        st.write("**Función simbólica:**")
        st.latex(sp.latex(pw_expr))

        a = float(sp.N(sp.sympify(tramos[0][1])))
        b = float(sp.N(sp.sympify(tramos[-1][2])))
        T = b - a
        f_np = sp.lambdify(x, pw_expr, "numpy")

        st.subheader("Resultados")
        st.write(f"Período: **T = {T:.6g}**")

        def f_wrapped(t):
            arr = np.array(t)
            try:
                return f_np(arr)
            except Exception:
                return np.vectorize(lambda s: float(f_np(s)))(arr)

        with st.spinner("Calculando coeficientes..."):
            A0, An, Bn = compute_coefficients(f_wrapped, a, b, N)

        st.write("### Coeficientes (numéricos)")
        st.write(f"a₀ = {A0:.6g}")
        for i in range(len(An)):
            st.write(f"a{i+1} = {An[i]:.6g},   b{i+1} = {Bn[i]:.6g}")

        # Energía
        energy_time = (1.0 / T) * integrate.quad(lambda t: float(f_wrapped(t)**2), a, b, limit=200)[0]
        energy_freq = A0**2 + 0.5 * np.sum(An**2 + Bn**2)

        st.write("### Energía (Parseval)")
        st.write(f"E(tiempo) = {energy_time:.6g}")
        st.write(f"E(coeficientes, hasta N) = {energy_freq:.6g}")
        st.write(f"Error relativo = {abs(energy_time - energy_freq)/max(1e-12, energy_time):.2%}")

        # Gráfica
        xs = np.linspace(a, b, resolution)
        ys = f_wrapped(xs)
        y_trunc = truncated_series_eval(A0, An, Bn, a, b, xs)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(xs, ys, label="f(x) original")
        ax.plot(xs, y_trunc, "--", label=f"Serie truncada (N={N})")
        ax.legend()
        ax.grid(True)
        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")
        st.pyplot(fig)

        st.success("✅ Cálculo completado")



    else:
        st.info("Complete los tramos y presione **Calcular** para obtener la serie de Fourier.")

    
elif section == "Parte II - FFT e Imágenes":
    st.title("Parte II – Filtros en Imágenes (FFT)")

    uploaded = st.file_uploader("Sube una imagen", type=["jpg", "png", "jpeg"])

    filtro_tipo = st.selectbox("Tipo de filtro", ["Pasa bajas", "Pasa altas"])
    f_cut = st.slider("Frecuencia de corte (f_cutoff)", 0.0, 1.0, 0.3, 0.01)

    if uploaded:
        # --- Leer imagen y convertir a float
        img = cv2.imdecode(np.frombuffer(uploaded.read(), np.uint8), cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


        # --- FFT por canal (R, G, B)
        filtered_channels = []
        magnitude_spectrum = None
        rows, cols, _ = img_rgb.shape
        crow, ccol = rows // 2, cols // 2
        mask = np.zeros((rows, cols, 3), np.float32)

        # --- Crear la máscara (pasa bajas / pasa altas)
        radius = int(f_cut * min(crow, ccol))
        Y, X = np.ogrid[:rows, :cols]
        dist = np.sqrt((X - ccol)**2 + (Y - crow)**2)
        if filtro_tipo == "Pasa bajas":
            mask = np.repeat((dist <= radius).astype(np.float32)[:, :, np.newaxis], 3, axis=2)

        else:
            mask = np.repeat((dist > radius).astype(np.float32)[:, :, np.newaxis], 3, axis=2)


        # --- Aplicar FFT a cada canal
        for i in range(3):
            F = np.fft.fft2(img_rgb[:, :, i])
            F_shift = np.fft.fftshift(F)
            F_filtered = F_shift * mask[:, :, i]
            F_ishift = np.fft.ifftshift(F_filtered)
            img_back = np.abs(np.fft.ifft2(F_ishift))
            img_back = np.clip(img_back / np.max(img_back), 0, 1)
            filtered_channels.append(img_back)

            if i == 0:  # mostrar espectro solo una vez
                magnitude_spectrum = np.log(1 + np.abs(F_shift))
                magnitude_spectrum = magnitude_spectrum / np.max(magnitude_spectrum)

        # --- Combinar canales
        img_filtered = np.dstack(filtered_channels)
        img_filtered = np.clip(img_filtered, 0, 1)

        col1, col2, col3 = st.columns(3)
        col1.image(img_rgb, caption="Original", use_container_width=True)
        col2.image(magnitude_spectrum, caption="Espectro (normalizado)", use_container_width=True, clamp=True)
        col3.image(img_filtered, caption=f"Filtrada ({filtro_tipo}, f_c={f_cut})", use_container_width=True, clamp=True)

        # --- Análisis automático
        st.markdown("### 💡 Análisis:")
        if filtro_tipo == "Pasa bajas":
            st.info("El filtro pasa bajas elimina las altas frecuencias → la imagen se suaviza y pierde detalles finos.")
        else:
            st.info("El filtro pasa altas elimina las bajas frecuencias → la imagen resalta bordes y detalles, pero pierde regiones suaves.")