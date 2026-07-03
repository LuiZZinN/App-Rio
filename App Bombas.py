# -*- coding: utf-8 -*-
"""
Ferramenta Integrada de Sizing e Validação Analítica/CFD de Rotores Centrifugos
Desenvolvido por: Engenheiro de Turbomáquinas Sênior & Full-Stack Developer
"""

import streamlit as st
import math
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# CONFIGURAÇÕES DA PÁGINA DO STREAMLIT
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Dimensionamento de Rotores - Centrifugal Pumps",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilo CSS para emular uma plataforma profissional de engenharia
st.markdown("""
<style>
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #1e293b;
    }
    .metric-label {
        font-size: 13px;
        color: #64748b;
        margin-top: 4px;
    }
    .cfd-box {
        background-color: #0f172a;
        color: #e2e8f0;
        font-family: 'Courier New', Courier, monospace;
        padding: 20px;
        border-radius: 8px;
        border-left: 5px solid #3b82f6;
    }
</style>
""", unsafe_allow_html=True)

# CONSTANTES DE ENGENHARIA
G_ACCEL = 9.81  # m/s^2
DENSITY_WATER = 1000.0  # kg/m^3

# -----------------------------------------------------------------------------
# FUNÇÕES DE CÁLCULO E TRIGONOMETRIA DOS TRIÂNGULOS DE VELOCIDADES
# -----------------------------------------------------------------------------
def calcular_especifica_metric(N, Q_m3s, H):
    """Calcula a velocidade específica de rotação n_q."""
    if H <= 0:
        return 0
    return N * (math.sqrt(Q_m3s)) / (H ** 0.75)

def classificar_tipo_rotor(nq):
    """Classifica o tipo do rotor baseado na velocidade específica n_q."""
    if nq <= 0:
        return "N/A"
    elif nq < 30:
        return "Radial Lento (Canais estreitos, alta carga)"
    elif nq < 50:
        return "Radial Normal"
    elif nq < 80:
        return "Radial Rápido"
    elif nq < 150:
        return "Fluxo Misto (Semi-axial)"
    else:
        return "Fluxo Axial"

def desenhar_triangulo(u, cm, cu, beta_deg, title, color_u="#1d4ed8", color_w="#eab308", color_c="#dc2626"):
    """
    Gera uma figura Matplotlib contendo a representação gráfica do triângulo de velocidades.
    Vetores:
      - U: velocidade tangencial/periférica (base horizontal)
      - Cm: componente meridiana perpendicular à base
      - Cu: projeção tangencial da velocidade absoluta
      - C: velocidade absoluta do fluido (soma de U e W)
      - W: velocidade relativa do fluido
    """
    fig, ax = plt.subplots(figsize=(6, 3))
    
    # Desenho dos vetores baseado nas coordenadas de origem (0,0)
    # Vetor U (Velocidade Periférica) - Horizontal
    ax.annotate("", xy=(u, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color=color_u, lw=2.5, mutation_scale=15),
                label=f"U = {u:.2f} m/s (Arraste)")
    
    # Ponto final de C é (Cu, Cm)
    # Vetor C (Velocidade Absoluta)
    ax.annotate("", xy=(cu, cm), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color=color_c, lw=2.5, mutation_scale=15),
                label=f"C = {math.sqrt(cm**2 + cu**2):.2f} m/s (Absoluta)")
    
    # Vetor W (Velocidade Relativa) conecta o final de U(u,0) ao final de C(cu, cm)
    ax.annotate("", xy=(cu, cm), xytext=(u, 0),
                arrowprops=dict(arrowstyle="->", color=color_w, lw=2.5, mutation_scale=15),
                label=f"W = {math.sqrt(cm**2 + (u-cu)**2):.2f} m/s (Relativa)")
    
    # Marcações e anotações visuais
    ax.plot([cu, cu], [0, cm], "k--", alpha=0.5) # Linha pontilhada da componente Cm
    ax.text(u * 0.5, -0.15 * cm, "U", color=color_u, fontsize=12, fontweight="bold", ha="center")
    ax.text(cu * 0.4, cm * 0.5, "C", color=color_c, fontsize=12, fontweight="bold", ha="right")
    ax.text((u + cu)*0.5, cm * 0.6, "W", color=color_w, fontsize=12, fontweight="bold", ha="left")
    
    # Angulações
    ax.text(cu + (u-cu)*0.1, cm * 0.1, f"β = {beta_deg:.1f}°", color="black", fontsize=10, fontweight="bold")
    
    # Determinação dos limites dos eixos com folga de segurança para visualização
    max_x = max(u, cu, 1.0)
    ax.set_xlim(-0.15 * max_x, 1.2 * max_x)
    ax.set_ylim(-0.2 * cm, 1.3 * cm)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    
    return fig

# -----------------------------------------------------------------------------
# INTERFACE PRINCIPAL & BARRA LATERAL (INPUTS)
# -----------------------------------------------------------------------------
st.title("⚙️ RotoPump: Sizing & CFD Centrifugal Pump Analyzer")
st.markdown("Plataforma Engenharia de Turbomáquinas de Alta Fidelidade para dimensionamento, análises de triângulos de velocidades e pós-processamento de CFD.")

# Sidebar de Parâmetros
st.sidebar.header("📊 Parâmetros de Entrada")

# Rádio para Modo de Funcionamento
modo = st.sidebar.radio(
    "Modo de Operação:",
    ("PROJETO (Sizing Técnico)", "VALIDAÇÃO/ANÁLISE (Geometria Fixa)")
)

st.sidebar.subheader("💧 Ponto de Operação")
Q_h = st.sidebar.number_input("Vazão Volumétrica Q (m³/h)", min_value=1.0, max_value=5000.0, value=60.0, step=5.0)
N_rpm = st.sidebar.number_input("Rotação Nominal N (rpm)", min_value=300.0, max_value=8000.0, value=1750.0, step=100.0)
density = st.sidebar.number_input("Densidade do Fluido ρ (kg/m³)", min_value=1.0, max_value=2000.0, value=1000.0, step=10.0)

# Conversões Importantes
Q_m3s = Q_h / 3600.0
omega = (N_rpm * 2 * math.pi) / 60.0

if modo == "PROJETO (Sizing Técnico)":
    st.sidebar.subheader("📐 Premissas de Projeto")
    H_req = st.sidebar.number_input("Carga Requerida H (m)", min_value=1.0, max_value=400.0, value=35.0, step=1.0)
    cm1_req = st.sidebar.slider("Cm Entrada cm1 (m/s)", 1.0, 12.0, 4.0, 0.1)
    cm2_req = st.sidebar.slider("Cm Saída cm2 (m/s)", 1.0, 12.0, 3.5, 0.1)
    beta2_blade_req = st.sidebar.slider("Ângulo de Saída da Pá β₂ (graus)", 15.0, 40.0, 22.5, 0.5)
    eta_est = st.sidebar.number_input("Eficiência Hidráulica Est. (η_h)", min_value=0.5, max_value=0.98, value=0.85, step=0.01)
    psi1 = st.sidebar.slider("Fator Obstrução Entrada (ψ₁)", 0.80, 1.0, 0.90, 0.01)
    psi2 = st.sidebar.slider("Fator Obstrução Saída (ψ₂)", 0.80, 1.0, 0.92, 0.01)
    z_blades = st.sidebar.slider("Número de Pás (z)", 3, 10, 6, 1)

else:
    st.sidebar.subheader("📐 Geometria Fixa Conhecida")
    D1_fixed = st.sidebar.number_input("Diâmetro de Entrada D₁ (mm)", min_value=10.0, max_value=600.0, value=108.0, step=1.0)
    D2_fixed = st.sidebar.number_input("Diâmetro de Saída D₂ (mm)", min_value=20.0, max_value=1200.0, value=250.0, step=2.0)
    b1_fixed = st.sidebar.number_input("Largura de Canal Entrada b₁ (mm)", min_value=1.5, max_value=200.0, value=18.0, step=0.5)
    b2_fixed = st.sidebar.number_input("Largura de Canal Saída b₂ (mm)", min_value=1.0, max_value=150.0, value=9.5, step=0.5)
    beta1_blade_fixed = st.sidebar.number_input("Ângulo Entrada da Pá β₁_blade (~)", min_value=5.0, max_value=45.0, value=18.0, step=0.5)
    beta2_blade_fixed = st.sidebar.number_input("Ângulo Saída da Pá β₂_blade (~)", min_value=10.0, max_value=60.0, value=22.5, step=0.5)
    
    st.sidebar.subheader("🌀 Efeitos de Fluidodinâmica")
    use_slip = st.sidebar.checkbox("Aplicar Correção de Deslizamento (Slip Factor)", value=True)
    z_blades = st.sidebar.slider("Número de Pás (z)", 3, 12, 6, 1)
    psi1 = st.sidebar.slider("Fator Obstrução Entrada (ψ₁)", 0.80, 1.0, 0.90, 0.01)
    psi2 = st.sidebar.slider("Fator Obstrução Saída (ψ₂)", 0.80, 1.0, 0.92, 0.01)

# -----------------------------------------------------------------------------
# NÚCLEO MATEMÁTICO DO MOTOR DE CÁLCULO
# -----------------------------------------------------------------------------
try:
    if modo == "PROJETO (Sizing Técnico)":
        # Altura teórica ideal Euler de projeto
        H_euler = H_req / eta_est
        
        # Sizing de U2 baseada em Euler simplificado com Cu1 = 0 (Escoamento puramente radial)
        # g * H_euler = U2 * (U2 - (cm2 / tan(beta2)))
        # U2^2 - U2 * (cm2 / tan(beta2)) - g * H_euler = 0 (Equação quadrática)
        beta2_rad = math.radians(beta2_blade_req)
        term_b = cm2_req / math.tan(beta2_rad)
        term_c = -G_ACCEL * H_euler
        
        # Raiz positiva da equação quadrática
        U2 = 0.5 * (term_b + math.sqrt(term_b**2 - 4 * term_c))
        D2 = (2 * U2) / omega
        b2 = Q_m3s / (math.pi * D2 * cm2_req * psi2)
        
        # Determinação de D1 e b1 baseados na Velocidade específica n_q
        nq = calcular_especifica_metric(N_rpm, Q_m3s, H_req)
        # Empiric ratios normais em bombas
        ratio_d1_d2 = max(0.35, min(0.70, 0.35 + 0.0015 * nq))
        D1 = ratio_d1_d2 * D2
        U1 = (omega * D1) / 2
        
        # Largura da entrada b1 baseada na Continuidade
        b1 = Q_m3s / (math.pi * D1 * cm1_req * psi1)
        
        # Ângulos cinemáticos das pás do escoamento ideal (shockless)
        beta1_rad = math.atan(cm1_req / U1)
        beta1_blade = math.degrees(beta1_rad)
        beta2_blade = beta2_blade_req
        
        # Vetores Cinéticos Entrada (Inlet) - Cu1 = 0
        U1_val, Cm1_val, Cu1_val, W1_val, C1_val = U1, cm1_req, 0.0, math.sqrt(cm1_req**2 + U1**2), cm1_req
        beta1_fluid_deg = beta1_blade
        
        # Vetores Cinéticos Saída (Outlet)
        U2_val, Cm2_val = U2, cm2_req
        Cu2_val = U2 - (cm2_req / math.tan(beta2_rad))
        C2_val = math.sqrt(Cm2_val**2 + Cu2_val**2)
        W2_val = math.sqrt(Cm2_val**2 + (U2 - Cu2_val)**2)
        beta2_fluid_deg = beta2_blade
        
        # Torque Teórico e Carga Gerada
        torque_euler = density * Q_m3s * (D2/2) * Cu2_val
        head_euler_calc = (U2 * Cu2_val) / G_ACCEL
        
    else:
        # Modo Validação: Geometria de Entrada é fixa, analisar operacionalmente
        D1 = D1_fixed / 1000.0
        D2 = D2_fixed / 1000.0
        b1 = b1_fixed / 1000.0
        b2 = b2_fixed / 1000.0
        beta1_blade = beta1_blade_fixed
        beta2_blade = beta2_blade_fixed
        
        # Velocidades Periféricas das Pás
        U1 = (omega * D1) / 2
        U2 = (omega * D2) / 2
        
        # Velocidades Meridianas a partir da Continuidade Física
        cm1_req = Q_m3s / (math.pi * D1 * b1 * psi1)
        cm2_req = Q_m3s / (math.pi * D2 * b2 * psi2)
        
        # Entrada: Assumindo Cu1 = 0
        U1_val, Cm1_val, Cu1_val = U1, cm1_req, 0.0
        C1_val = cm1_req
        W1_val = math.sqrt(cm1_req**2 + U1**2)
        beta1_fluid_deg = math.degrees(math.atan(cm1_req / U1))
        
        # Saída: Análise do triângulo de velocidades com ou sem Slip Factor
        beta2_blade_rad = math.radians(beta2_blade)
        
        if use_slip:
            # Fórmula empírica de Wiesner para deslizamento (Slip Factor)
            sigma_slip = 1.0 - (math.sqrt(math.sin(beta2_blade_rad)) / (z_blades ** 0.7))
            sigma_slip = max(0.5, min(1.0, sigma_slip))
        else:
            sigma_slip = 1.0
            
        # Cu2 levando em consideração o escoamento guiado
        Cu2_val = sigma_slip * U2 - (cm2_req / math.tan(beta2_blade_rad))
        
        if Cu2_val < 0:
            st.error("⚠️ Alerta Geral: As velocidades e ângulos geométricos informados indicam refluxo ou operação turbina indesejada (Cu2 < 0). Verifique suas dimensões geométricas.")
            Cu2_val = 0.01
            
        U2_val, Cm2_val = U2, cm2_req
        C2_val = math.sqrt(Cm2_val**2 + Cu2_val**2)
        W2_val = math.sqrt(Cm2_val**2 + (U2 - Cu2_val)**2)
        
        # Ângulo cinemático de saída do fluido realizado
        beta2_fluid_rad = math.atan(Cm2_val / (U2 - Cu2_val)) if (U2 - Cu2_val) != 0 else math.pi/2
        beta2_fluid_deg = math.degrees(beta2_fluid_rad)
        
        # Carga correspondente de Euler idealizada
        head_euler_calc = (U2 * Cu2_val) / G_ACCEL
        torque_euler = density * Q_m3s * (D2/2) * Cu2_val
        nq = calcular_especifica_metric(N_rpm, Q_m3s, head_euler_calc)

    # -----------------------------------------------------------------------------
    # APRESENTAÇÃO DOS RESULTADOS (TABS E COLUNAS)
    # -----------------------------------------------------------------------------
    aba1, aba2, aba3, aba4 = st.tabs([
        "📊 Resultados e Desempenho", 
        "📐 Triângulos de Velocidades", 
        "💻 Bloco de Notas para CFD/CAD",
        "🎯 Validação e Pós-Processamento CFD"
    ])
    
    with aba1:
        st.subheader("📌 Resumo Dimensional do Rotor Sizing")
        
        # Métricas de destaque
        m1, m2, m3, m4, m5 = st.columns(5)
        
        m1.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{D1*1000:.1f} mm</div>
            <div class="metric-label">Diâmetro Entrada (D₁)</div>
        </div>
        """, unsafe_allow_html=True)
        
        m2.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{D2*1000:.1f} mm</div>
            <div class="metric-label">Diâmetro Saída (D₂)</div>
        </div>
        """, unsafe_allow_html=True)
        
        m3.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{b1*1000:.1f} mm</div>
            <div class="metric-label">Largura Entrada (b₁)</div>
        </div>
        """, unsafe_allow_html=True)
        
        m4.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{b2*1000:.1f} mm</div>
            <div class="metric-label">Largura Saída (b₂)</div>
        </div>
        """, unsafe_allow_html=True)
        
        m5.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{beta2_blade:.1f}°</div>
            <div class="metric-label">Ângulo da Pá β₂</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Colunas com tabelas de velocidades cinemáticas
        c1, c2 = st.columns(2)
        
        with c1:
            st.markdown("### 📥 Cinemática de Entrada (Inlet)")
            data_inlet = {
                "Parâmetro Cinemático": [
                    "Velocidade Tangencial (U₁)",
                    "Velocidade Meridiana (Cm₁)",
                    "Velocidade Absoluta (C₁)",
                    "Velocidade Relativa (W₁)",
                    "Ângulo Operacional do Fluido (β₁)"
                ],
                "Valor": [
                    f"{U1_val:.2f} m/s",
                    f"{Cm1_val:.2f} m/s",
                    f"{C1_val:.2f} m/s",
                    f"{W1_val:.2f} m/s",
                    f"{beta1_fluid_deg:.1f}°"
                ],
                "Fórmula": [
                    "U = ω * D / 2",
                    "Cm = Q / (π * D * b * ψ)",
                    "C = √(Cm² + Cu²)",
                    "W = √(Cm² + (U - Cu)²)",
                    "β = atan(Cm / U)"
                ]
            }
            st.table(data_inlet)
            
        with c2:
            st.markdown("### 📤 Cinemática de Saída (Outlet)")
            data_outlet = {
                "Parâmetro Cinemático": [
                    "Velocidade Tangencial (U₂)",
                    "Velocidade Meridiana (Cm₂)",
                    "Velocidade Tangencial Fluid (Cu₂)",
                    "Velocidade Absoluta (C₂)",
                    "Velocidade Relativa (W₂)",
                    "Ângulo Operacional Realizado (β₂)"
                ],
                "Valor": [
                    f"{U2_val:.2f} m/s",
                    f"{Cm2_val:.2f} m/s",
                    f"{Cu2_val:.2f} m/s",
                    f"{C2_val:.2f} m/s",
                    f"{W2_val:.2f} m/s",
                    f"{beta2_fluid_deg:.1f}°"
                ],
                "Fórmula": [
                    "U = ω * D / 2",
                    "Cm = Q / (π * D * b * ψ)",
                    "Cu = σ * U - Cm / tan(β)",
                    "C = √(Cm² + Cu²)",
                    "W = √(Cm² + (U - Cu)²)",
                    "β = atan(Cm / (U - Cu))"
                ]
            }
            st.table(data_outlet)
            
        st.subheader("💡 Características de Desempenho Teórico")
        metrics_perf = st.columns(4)
        
        # Potência Hidráulica útil de Fluido (estimado)
        P_fluid = (density * G_ACCEL * Q_m3s * (H_req if modo == "PROJETO (Sizing Técnico)" else head_euler_calc * 0.82)) / 1000.0
        P_euler = (density * G_ACCEL * Q_m3s * head_euler_calc) / 1000.0
        
        metrics_perf[0].metric("Velocidade Específica (nq)", f"{nq:.1f}", classificar_tipo_rotor(nq))
        metrics_perf[1].metric("Carga Ideal Euler (H_euler)", f"{head_euler_calc:.2f} m")
        metrics_perf[2].metric("Torque de Euler Teórico", f"{torque_euler:.1f} N.m")
        metrics_perf[3].metric("Potência Teórica Fluid", f"{P_euler:.2f} kW")

    with aba2:
        st.subheader("📐 Renderização dos Triângulos de Velocidades")
        st.info("O fechamento geométrico obedece puramente às equações de turbomáquinas sob escoamento radial sem turbulência.")
        
        fig_cols = st.columns(2)
        with fig_cols[0]:
            fig_in = desenhar_triangulo(U1_val, Cm1_val, Cu1_val, beta1_fluid_deg, "Triângulo de Entrada (Inlet)")
            st.pyplot(fig_in)
            st.write("**Entrada:** Como Cu₁ = 0, C₁ é perfeitamente perpendicular à velocidade tangencial U₁.")
            
        with fig_cols[1]:
            fig_out = desenhar_triangulo(U2_val, Cm2_val, Cu2_val, beta2_fluid_deg, "Triângulo de Saída (Outlet)")
            st.pyplot(fig_out)
            st.write("**Saída:** Mostra a deflexão causada pelo bordo de fuga das pás, impulsionando e energizando o fluido.")

    with aba3:
        st.subheader("🖥️ Parâmetros prontos para ANSYS SpaceClaim (CAD) & Fluent (CFD)")
        st.markdown("Copie as especificações abaixo diretamente para parametrizar seu CAD tridimensional e condições de contorno de CFD do solver ANSYS Fluent.")
        
        cfd_text = f"""========================================================================
1. PARÂMETROS DE GEOMETRIA (SpaceClaim / DesignModeler)
================================----------------------------------------
- Diâmetro de Entrada (D1):        {D1*1000:.2f} mm
- Diâmetro de Saída (D2):          {D2*1000:.2f} mm
- Largura do Canal Entrada (b1):   {b1*1000:.2f} mm
- Largura do Canal Saída (b2):     {b2*1000:.2f} mm
- Ângulo de Entrada de Projeto (β1): {beta1_blade:.2f}°
- Ângulo de Saída de Projeto (β2):  {beta2_blade:.2f}°
- Número Total de Pás (z):          {z_blades}

========================================================================
2. CONDIÇÕES DE CONTORNO DO FLUIDO (ANSYS Fluent / CFX Setup)
================================----------------------------------------
- Velocidade Angular do Rotor (ω):  {omega:.3f} rad/s (Rotação: {N_rpm} rpm)
- Vazão Mássica na Entrada (Mass Flow Rate): {(Q_m3s * density):.4f} kg/s (Density: {density} kg/m³)
- Condição de Entrada (Inlet):      Mass Flow Inlet ({density * Q_m3s:.3f} kg/s)
- Condição de Saída (Outlet):       Pressure Outlet (0 Pa gauge)
- Zona Móvel (MRF / Sliding Mesh):  Rotational Frame Speed = {-omega:.3f} rad/s (Z-axis)

========================================================================
3. VALORES TEÓRICOS DE REFERÊNCIA PARA VALIDAÇÃO CFD
================================----------------------------------------
- Carga Teórica de Euler (H_ideal): {head_euler_calc:.2f} m (Carga total líquida esperada)
- Torque Hidráulico Esperado de Euler: {torque_euler:.2f} N.m
- Potência Euleriana:               {P_euler:.2f} kW
========================================================================"""
        
        st.text_area("Bloco de Configuração ANSYS", value=cfd_text, height=450)
        st.success("✔️ Parâmetros consistentes calculados analiticamente!")

    with aba4:
        st.subheader("🎯 Seção de Pós-Processamento e Correlação de Validação")
        st.markdown("Insira aqui os valores numéricos coletados em sua simulação Fluent convergida para obter o relatório de Eficiência Hidráulica de correlação.")
        
        post_cols = st.columns(2)
        with post_cols[0]:
            h_sim = st.number_input("Carga Manométrica Simulada H_sim (m)", min_value=0.1, max_value=500.0, value=max(1.0, head_euler_calc * 0.8), step=0.5)
            t_sim = st.number_input("Torque Simulado na Turbina T_sim (N.m)", min_value=0.01, max_value=20000.0, value=max(0.1, torque_euler * 1.1), step=0.1)
        
        with post_cols[1]:
            # Cálculos de pós-processamento
            p_fluid_sim = (density * G_ACCEL * Q_m3s * h_sim) / 1000.0 # kW
            p_shaft_sim = (t_sim * omega) / 1000.0 # kW (T * omega)
            
            if p_shaft_sim > 0:
                eta_h_sim = (p_fluid_sim / p_shaft_sim) * 100.0
            else:
                eta_h_sim = 0
                
            dev_head = ((h_sim - head_euler_calc) / head_euler_calc) * 100.0 if head_euler_calc > 0 else 0
            dev_torque = ((t_sim - torque_euler) / torque_euler) * 100.0 if torque_euler > 0 else 0
            
            st.markdown("### 🏆 CFD vs Solução Analítica")
            st.write(f"- **Eficiência Hidráulica Simulada (η_h):** \`{eta_h_sim:.2f}%\`")
            st.write(f"- **Desvio de Carga (Contra Euler):** \`{dev_head:+.1f}%\` (Simulado vs Euler Teórico)")
            st.write(f"- **Desvio de Torque:** \`{dev_torque:+.1f}%\` (Desvio no torque de Euler)")
            
            # Barras visuais de Eficiência
            st.write("Medição de Eficiência Hidráulica:")
            st.progress(min(1.0, max(0.01, eta_h_sim / 100.0)))
            
            if eta_h_sim > 90:
                st.warning("⚠️ Eficiência acima de 90% indica um modelo sem perdas viscosas acopladas; verifique seu modelo de turbulência no Fluent.")
            elif eta_h_sim < 60:
                st.info("💡 Eficiência abaixo de 60% sugere altas perdas por descolamento de camada limite ou perdas volumétricas. Ajuste os ângulos de ataque.")
            else:
                st.success("🎯 Eficiência na faixa saudável para bombas industriais de escoamento radial pura!")

except Exception as e:
    st.error(f"❌ Ocorreu um erro matemático ao tentar harmonizar os inputs: {e}. Certifique-se de que nenhum divisor seja zero nas simplificações.")
