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

st.title("Map Serie de Fourier Truncada y FFT")

section = st.sidebar.radio("Selecciona la sección:", 
                           ["Parte I - Series de Fourier", 
                            "Parte II - FFT e Imágenes"])

if section == "Parte I - Series de Fourier":
    
    st.set_page_config(page_title="Series de Fourier - Función por tramos ", layout="wide")
    st.title("Series de Fourier: Funciones por tramos de 2N + 1 términos")

    st.markdown("""
    **Instrucciones:**
    1. Ingrese cuántos tramos tiene la función.
    2. Especifique para cada tramo su expresión y el intervalo donde aplica.
    3. Ajuste el número de armónicos (N) y presione **Calcular**.
    """)

    # --- Sidebar Inputs ---
    st.sidebar.header("Entrada de datos")

    num_tramos = st.sidebar.number_input("Número de tramos", min_value=1, max_value=10, value=3, step=1)

    # Guardamos los tramos en una lista
    tramos = []
    for i in range(num_tramos):
        st.sidebar.markdown(f"**Tramo {i+1}:**")
        expr_text = st.sidebar.text_input(f"f{i+1}(x) =", i+1 , key=f"expr_{i}")
        a_i = st.sidebar.text_input(f"Inicio (a{i+1})", "0", key=f"a_{i}")
        b_i = st.sidebar.text_input(f"Fin (b{i+1})", "0", key=f"b_{i}")
        tramos.append((expr_text, a_i, b_i))

    N = st.sidebar.slider("N (armónicos)", min_value=0, max_value=50, value=5)
    resolucion = st.sidebar.slider("Puntos para graficar", 200, 5000, 1000)
    calcular = st.sidebar.button("Calcular")

    x = sp.symbols('x')

    # --- Funciones auxiliares ---
    def construir_piecewise(tramos):
        """
        Construye una expresión SymPy Piecewise a partir de los tramos ingresados.
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

    def calcular_coeficientes(f_np, a, b, N):
        T = b - a
        A0 = (1.0 / T) * integrar(f_np, a, b)
        An, Bn = [], []
        for n in range(1, N + 1):
            cos_fun = lambda t, n=n: f_np(t) * np.cos(2 * np.pi * n * (t - a) / T) #Funcion coseno f(t) * cos(...)
            sin_fun = lambda t, n=n: f_np(t) * np.sin(2 * np.pi * n * (t - a) / T) #Funcion seno   f(t) * sin(...)
            an = (2.0 / T) * integrate.quad(lambda t: float(cos_fun(t)), a, b, limit=200)[0] 
            bn = (2.0 / T) * integrate.quad(lambda t: float(sin_fun(t)), a, b, limit=200)[0]
            An.append(an)
            Bn.append(bn)
        return A0, np.array(An), np.array(Bn)

    def evaluar_serie_truncada(A0, An, Bn, a, b, xs): #Evalua la serie truncada en los puntos x
        T = b - a
        y = np.full_like(xs, A0, dtype=float)
        for idx, n in enumerate(range(1, len(An) + 1)):
            y += An[idx] * np.cos(2 * np.pi * n * (xs - a) / T) + Bn[idx] * np.sin(2 * np.pi * n * (xs - a) / T)
        return y

    # --- Main computation ---
    if calcular:
        expr_partes = construir_piecewise(tramos)
        if expr_partes is None:
            st.stop()
        
        st.write("**Función por partes:**")
        st.latex(sp.latex(expr_partes))

        a = float(sp.N(sp.sympify(tramos[0][1]))) #Primer límite
        b = float(sp.N(sp.sympify(tramos[-1][2]))) #Último límite
        T = b - a #Período
        f_np = sp.lambdify(x, expr_partes, "numpy") 

        st.subheader("Resultados")
        st.write(f"Período: **T = {T:.6g}**")

        def f_wrapped(t): #Estandariza la entrada para que siempre sea un array numpy 
            arr = np.array(t)
            return f_np(arr)


        with st.spinner("Calculando coeficientes..."):
            A0, An, Bn = calcular_coeficientes(f_wrapped, a, b, N)

        st.write("### Coeficientes (numéricos)")
        st.write(f"a₀ = {A0:.6g}")
        for i in range(len(An)):
            st.write(f"a{i+1} = {An[i]:.6g},   b{i+1} = {Bn[i]:.6g}")

        # Energía
        energia = integrate.quad(lambda t: float(f_wrapped(t)**2), a, b, limit=200)[0]
        ice_parentesis = (T * (A0**2 + 0.5 * np.sum(An**2 + Bn**2)))
        ice= energia - ice_parentesis

        st.write("### Energía")
        st.write(f"Energia de la señal = {energia:.6g}")
        st.write(f"Energia según coeficientes (ICE) = {ice_parentesis:.6g}  ")
        st.write(f"Integral cuadrada del Error (ICE) = {ice:.6g}  ")

        if ice <= 0.02 * energia:
            st.success("Cumple la condición de energía (ICE ≤ 0.02Ef)")
        else:
            st.error("No cumple la condición de energía (ICE ≤ 0.02Ef)")
        
        # Gráfica
        xs = np.linspace(a, b, resolucion)
        ys = f_wrapped(xs)

        y_trunc = evaluar_serie_truncada(A0, An, Bn, a, b, xs)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.set_title(f"Comparación: f(x) original vs Serie truncada (N={N})")
        ax.plot(xs, ys, label="f(x) original")
        ax.plot(xs, y_trunc, "--", label=f"Serie truncada (N={N})")
        ax.legend()
        ax.grid(True)
        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")
        st.pyplot(fig)
        
        #Ecuación de la serie truncada
        st.subheader("Ecuación de la serie de Fourier truncada")
        omega0 = 2 * sp.pi / T
        ecuacion = A0

        for n in range(1, N+1):
            ecuacion += An[n-1]*sp.cos(n*omega0*x) + Bn[n-1]*sp.sin(n*omega0*x)

        st.latex(ecuacion)
        

    else:
        st.info("Complete los tramos y presione **Calcular** para obtener la serie de Fourier.")

    
elif section == "Parte II - FFT e Imágenes":
    st.title("Parte II – Filtros en Imágenes (FFT)")

    uploaded = st.file_uploader("Sube una imagen", type=["jpg", "png", "jpeg"])

    filtro_tipo = st.selectbox("Tipo de filtro", ["Pasa bajas", "Pasa altas"])
    frecuencia_corte = st.slider("Frecuencia de corte (f_cutoff)", 0.0, 1.0, 0.3, 0.01)

    if uploaded:
        # --- Leer imagen y convertir a float
        img = cv2.imdecode(np.frombuffer(uploaded.read(), np.uint8), cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


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