# app.py
"""
Streamlit app - Parte 1: Serie de Fourier Truncada
Autor: (Generado por asistente). Ajusta el nombre si lo deseas.
Idioma: español
"""
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64

st.set_page_config(page_title="Serie de Fourier Truncada", layout="wide")

# ---------------------------
# Helpers
# ---------------------------
def safe_eval_fun(expr: str):
    """
    Devuelve una función vectorizada f(t) evaluando la expresión expr
    usando sólo funciones de numpy permitidas.
    """
    # mapa de nombres permitidos
    allowed = {
        'np': np,
        'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
        'exp': np.exp, 'sqrt': np.sqrt, 'abs': np.abs,
        'pi': np.pi, 'e': np.e,
        'arcsin': np.arcsin, 'arccos': np.arccos, 'arctan': np.arctan,
        'sinh': np.sinh, 'cosh': np.cosh, 'tanh': np.tanh,
        'sign': np.sign
    }
    # crear la función
    def f(t):
        # convertir a array numpy
        tt = np.array(t, dtype=float)
        # evaluación segura usando eval con ambiente controlado
        return eval(expr, {"__builtins__": {}}, {**allowed, 't': tt})
    return np.vectorize(f, otypes=[float])

def compute_coeffs(func, a, b, N, M=2000):
    """
    Calcula a0, a_n, b_n para n=1..N usando integracion numerica (trapezoid)
    - func : función vectorizada
    - [a,b] : intervalo de un periodo
    - N : número de armónicos
    - M : número de puntos para la integración
    Retorna: a0, array(a_n), array(b_n), Ef
    """
    T = b - a
    omega = 2 * np.pi / T
    t = np.linspace(a, b, M, endpoint=False)  # malla
    y = func(t)
    # energía Ef = integral_0^T f^2(t) dt
    Ef = np.trapezoid(y**2, t)
    # a0
    a0 = (1.0 / T) * np.trapezoid(y, t)
    an = np.zeros(N)
    bn = np.zeros(N)
    for n in range(1, N+1):
        cosn = np.cos(n * omega * t)
        sinn = np.sin(n * omega * t)
        an[n-1] = (2.0 / T) * np.trapezoid(y * cosn, t)
        bn[n-1] = (2.0 / T) * np.trapezoid(y * sinn, t)
    return a0, an, bn, Ef, t, y

def fourier_series_eval(a0, an, bn, T, t_eval):
    """Evalúa la serie truncada en t_eval"""
    omega = 2 * np.pi / T
    y = np.full_like(t_eval, a0, dtype=float)
    for n, (a_n, b_n) in enumerate(zip(an, bn), start=1):
        y += a_n * np.cos(n * omega * t_eval) + b_n * np.sin(n * omega * t_eval)
    return y

def df_coeffs_to_csv(a0, an, bn):
    df = pd.DataFrame({
        'n': [0] + list(range(1, len(an)+1)),
        'a_n': [a0] + list(an),
        'b_n': [0] + list(bn)  # b0=0 as convención
    })
    return df

def get_csv_download_link(df, filename="coeficientes.csv"):
    buffer = BytesIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)
    b64 = base64.b64encode(buffer.read()).decode()
    href = f"data:file/csv;base64,{b64}"
    return href

# ---------------------------
# UI
# ---------------------------
st.title("Serie de Fourier Truncada — (Parte I)")

# left / right layout
col1, col2 = st.columns([1, 2])

with col1:
    st.header("Selección de función")
    option = st.selectbox("Función:", ["f1 (rectangular en [-2,2])", "f2 (rampa en [-2,2])", "Función personalizada"])
    st.markdown("**Intervalo por defecto para funciones predefinidas:** [-2, 2] (período T=4).")
    st.markdown("**Intervalo por defecto para personalizada:** [-π, π] (período T=2π).")
    if option == "f1 (rectangular en [-2,2])":
        st.markdown("Definida como:\n\n"
                    r"$$ f_1(t)=\begin{cases}0 & -2<t<-1\\ 1 & -1<t<1\\ 0 & 1<t<2\end{cases} $$")
        func_expr = None
        a = -2.0
        b = 2.0
    elif option == "f2 (rampa en [-2,2])":
        st.markdown("Definida como:\n\n"
                    r"$$ f_2(t)=\begin{cases}0 & -2<t<-1\\ t+2 & -1<t<1\\ 0 & 1<t<2\end{cases} $$")
        func_expr = None
        a = -2.0
        b = 2.0
    else:
        st.markdown("Ingrese la función periódica **en términos de `t`**. Use funciones de `numpy` (sin, cos, exp, sqrt, etc.). Ejemplo: `sin(t) + 0.5*cos(2*t)`")
        func_expr = st.text_input("Expresión f(t):", value="sin(t) + 0.5*cos(2*t)")
        a = st.number_input("Inicio del intervalo (a):", value=-np.pi, format="%.6f")
        b = st.number_input("Fin del intervalo (b):", value=np.pi, format="%.6f")
        if b <= a:
            st.error("El extremo b debe ser mayor que a.")
    st.markdown("---")
    st.header("Parámetros")
    N = st.slider("Número de armónicos N (la serie tendrá 2N+1 términos):", min_value=1, max_value=60, value=10)
    malla_points = st.number_input("Puntos para integración numérica (más -> más precisión):", min_value=200, max_value=20000, value=4000, step=200)
    st.markdown("Botones rápidos para mostrar N = 3, 5, 10:")
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

with col2:
    st.header("Resultado — visualización y datos")
    # Construir función seleccionada
    if option == "f1 (rectangular en [-2,2])":
        def f1_vec(t):
            t = np.array(t)
            # map t to principal interval [-2,2)
            # For series, we assume func defined in [-2,2)
            # Return values as piecewise
            y = np.zeros_like(t, dtype=float)
            # open intervals as in enunciado
            mask = (t > -1) & (t < 1)
            y[mask] = 1.0
            return y
        func = np.vectorize(f1_vec, otypes=[float])
        a_interval = -2.0
        b_interval = 2.0
    elif option == "f2 (rampa en [-2,2])":
        def f2_vec(t):
            t = np.array(t)
            y = np.zeros_like(t, dtype=float)
            mask = (t > -1) & (t < 1)
            y[mask] = t[mask] + 2.0
            return y
        func = np.vectorize(f2_vec, otypes=[float])
        a_interval = -2.0
        b_interval = 2.0
    else:
        # crear función segura a partir de expr
        try:
            func = safe_eval_fun(func_expr)
            a_interval = float(a)
            b_interval = float(b)
        except Exception as e:
            st.error(f"Error al crear la función: {e}")
            st.stop()

    # cálculo
    with st.spinner("Calculando coeficientes y series..."):
        a0, an, bn, Ef, t_malla, y_true = compute_coeffs(func, a_interval, b_interval, N, M=malla_points)
        T = b_interval - a_interval
        # eval de la serie en la malla
        y_approx = fourier_series_eval(a0, an, bn, T, t_malla)
        # ICE
        series_energy = a0**2 * T + (T / 2.0) * np.sum(an**2 + bn**2)
        ICE = Ef - series_energy
        ICE_ratio = ICE / Ef if Ef != 0 else np.nan
        condition_ok = ICE <= 0.02 * Ef if Ef != 0 else False

    # Mostrar datos calculados
    left, right = st.columns([1,1])
    with left:
        st.subheader("Datos del período y energía")
        st.write(f"Intervalo usado: [{a_interval:.6g}, {b_interval:.6g}]")
        st.write(f"Período T = {T:.6g}")
        st.write(f"Energía de la señal (Ef) ≈ {Ef:.6e} (integral numérica con {malla_points} puntos)")
        st.write(f"a₀ ≈ {a0:.6e}")
        st.write(f"ICE(N) ≈ {ICE:.6e}")
        st.write(f"ICE/Norma: ICE / Ef = {ICE_ratio:.6%}")
        st.success("ICE cumple  <= 0.02*Ef" if condition_ok else "ICE NO cumple  <= 0.02*Ef")
    with right:
        st.subheader("Parámetros de cálculo")
        st.write(f"Puntos integración (M): {malla_points}")
        st.write("Método: Regla del trapecio (numpy.trapezoid)")
        st.write("Evaluación de la serie con ω = 2π / T")
        st.write("Número de armónicos N = " + str(N))

    st.markdown("---")
    # Mostrar fórmula de la serie (LaTeX)
    st.subheader("Ecuación (serie trigonométrica truncada)")
    # Construir LaTeX de la serie truncada (breve)
    latex_terms = r"a_0 + " + " + ".join([f"a_{{{n}}}\\cos({n}\\omega t) + b_{{{n}}}\\sin({n}\\omega t)" for n in range(1, N+1)])
    st.latex(r"f_{N}(t) = " + latex_terms)

    st.markdown("---")
    # Graficas: original vs aproximacion
    st.subheader("Gráfica: señal original vs serie truncada (sobre un periodo)")
    fig, ax = plt.subplots(figsize=(8,3.5))
    ax.plot(t_malla, y_true, label="Original", linewidth=1.5)
    ax.plot(t_malla, y_approx, label=f"Serie truncada (N={N})", linestyle='--', linewidth=1.2)
    ax.set_xlabel("t")
    ax.set_ylabel("f(t)")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    st.markdown("---")
    # Mostrar residuo y error puntual
    st.subheader("Error puntual y residuo")
    fig2, ax2 = plt.subplots(figsize=(8,3.5))
    resid = y_true - y_approx
    ax2.plot(t_malla, resid)
    ax2.set_title("Residuo: f(t) - f_N(t)")
    ax2.set_xlabel("t")
    ax2.grid(True)
    st.pyplot(fig2)

    st.markdown("---")
    # Tabla de coeficientes
    st.subheader("Coeficientes calculados (primeros 20 mostrados)")
    df_coef = pd.DataFrame({
        'n': np.arange(0, N+1),
        'a_n': np.concatenate([[a0], an]),
        'b_n': np.concatenate([[0.0], bn])
    })
    st.dataframe(df_coef.head(20), use_container_width=True)

    # CSV download
    #csv_href = get_csv_download_link(df_coef, "coeficientes_fourier.csv")
    #st.markdown(f"[Descargar coeficientes (CSV)]({csv_href})")

    st.markdown("---")
    # Documentación paso a paso (solicitud del enunciado)
    st.subheader("Documentación (qué se calculó y cómo)")
    st.markdown("""
    - Se calculó la integral numérica sobre el intervalo seleccionado con la regla del trapecio (numpy.trapezoid).  
    - Valores mostrados: periodo T, coeficientes a0, a_n, b_n para n=1..N, energía Ef, ICE(N) y la comprobación de la condición ICE(N) ≤ 0.02 Ef.  
    - Malla utilizada para integración: {M} puntos.  
    - Método para evaluar la serie: suma directa de términos trigonométricos truncados.
    """.format(M=malla_points))

    st.markdown("---")
    # Sección para probar varios N y mostrar comportamiento de ICE
    st.subheader("Evolución del ICE con N (prueba rápida)")
    Ns = [1,2,3,5,10,20,40]
    Ns = [n for n in Ns if n <= 60]
    ICE_vals = []
    Ef_display = Ef
    for ntest in Ns:
        a0t, ant, bnt, Eft, _, _ = compute_coeffs(func, a_interval, b_interval, ntest, M= min(8000, malla_points))
        series_energy_t = a0t**2 * T + (T / 2.0) * np.sum(ant**2 + bnt**2)
        ICET = Eft - series_energy_t
        ICE_vals.append(ICET)
    fig3, ax3 = plt.subplots(figsize=(8,3))
    ax3.plot(Ns, ICE_vals, marker='o')
    ax3.axhline(0.02 * Ef_display, color='red', linestyle='--', label='0.02 * Ef')
    ax3.set_xlabel("N")
    ax3.set_ylabel("ICE(N)")
    ax3.set_title("Evolución numérica de ICE(N)")
    ax3.grid(True)
    ax3.legend()
    st.pyplot(fig3)

    st.markdown("---")
    # Preguntas 2 (documentar N=3,5,10)
    st.subheader("Documentación específica: N = 3, 5, 10 (valores completos)")
    doc_Ns = [3,5,10]
    for ndoc in doc_Ns:
        a0d, andd, bndd, Efd, td, yd = compute_coeffs(func, a_interval, b_interval, ndoc, M=malla_points)
        st.markdown(f"**Resultados para N = {ndoc}**")
        st.write(f"a0 = {a0d:.6e}  |  Ef ≈ {Efd:.6e}")
        df_tmp = pd.DataFrame({
            'n': np.arange(1, ndoc+1),
            'a_n': andd,
            'b_n': bndd
        })
        st.dataframe(df_tmp, use_container_width=True)

    st.markdown("---")
    # Sección final: consejos para reporte
    st.subheader("Sugerencias para el reporte (LaTeX)")
    st.markdown("""
    - Incluye una sección donde copies la ecuación de la serie y pegues las tablas exportadas.  
    - Adjunta las gráficas (original vs aproximada) para N = 3, 5, 10.  
    - Explica la elección de la malla y cómo afecta la precisión.  
    - Muestra la verificación numérica de ICE(N) ≤ 0.02 Ef (tabla/figura).  
    """)

st.markdown("---")
st.caption("Nota: Si ingresas una función personalizada, asegúrate de usar únicamente las funciones permitidas listadas en la ayuda. La evaluación se realiza en un entorno restringido para evitar ejecuciones inseguras.")
