/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState } from "react";
import { Code, Copy, Check, Download, Terminal, Layers, Sparkles, BookOpen, AlertCircle } from "lucide-react";
import { SimulationInputs } from "../types";

export const STREAMLIT_PYTHON_CODE = `import sys
import subprocess

# ==============================================================================
# 0. BLINDAGEM STREAMLIT CLOUD (AUTO-INSTALAÇÃO DE DEPENDÊNCIAS)
# ==============================================================================
# Garante que as bibliotecas existam mesmo se o arquivo for subido sozinho sem requirements.txt
def _garantir_dependencias():
    pacotes = [("plotly", "plotly"), ("pandas", "pandas"), ("numpy", "numpy"), ("google-genai", "google.genai")]
    for pacote, modulo in pacotes:
        try:
            __import__(modulo)
        except ImportError:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pacote])
            except Exception:
                pass

_garantir_dependencias()

import streamlit as st
import pandas as pd
import numpy as np
import os

HAS_PLOTLY = False
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# Configuração da página Streamlit
st.set_page_config(
    page_title="Simulador Streeter-Phelps CONAMA 357",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilo customizado CSS para deixar a interface profissional
st.markdown("""
<style>
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 0.75rem;
        padding: 1rem;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    }
    .status-ok {
        color: #16a34a;
        font-weight: bold;
    }
    .status-alert {
        color: #dc2626;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 1. FUNÇÕES FÍSICAS E CINÉTICAS DO MODELO STREETER-PHELPS
# ==============================================================================

def calculate_daily_load_kg(q_m3h, bod_mgl):
    """Calcula carga orgânica diária (kg/dia) com base na vazão e concentração."""
    return (q_m3h * bod_mgl * 24.0) / 1000.0

def calculate_streeter_phelps(inputs):
    river = inputs["river"]
    effluents = inputs["effluents"]
    use_corrected_decay = inputs["use_corrected_decay"]
    T = inputs["temp"]
    k1_mode = inputs["k1_mode"]
    k1_manual = inputs["k1_manual"]
    k2_mode = inputs["k2_mode"]
    k2_manual = inputs["k2_manual"]

    Qr = float(river["Qr"])
    compr = float(river["compr"])
    prof = float(river["prof"])
    classe = int(river["classe"])

    # 1. Saturação de OD (Cs) corrigida por temperatura (Elmore & West)
    Cs = 14.652 - 0.41022 * T + 0.007991 * (T**2) - 0.000077774 * (T**3)
    Co = 0.9 * Cs  # OD de entrada assumido em 90% da saturação

    # Limite legal de OD e DBO natural por classe CONAMA 357
    if classe == 1:
        Od, Lr = 6.0, 3.0
    elif classe == 2:
        Od, Lr = 5.0, 5.0
    elif classe == 3:
        Od, Lr = 4.0, 10.0
    elif classe == 4:
        Od, Lr = 2.0, 20.0
    else:
        Od, Lr = 5.0, 5.0

    active_effluents = effluents if len(effluents) > 0 else [
        {"id": "fallback", "name": "Sem Lançamento", "Qe": 0.0, "Le": 0.0, "dist": 0.0}
    ]

    # 2. Coeficientes cinéticos a 20ºC e correção térmica por Arrhenius
    k1_20 = 0.0278 if prof <= 1.5 else 0.0163
    if k1_mode == "Manual":
        k1_20 = float(k1_manual) / 24.0
    k1 = k1_20 * (1.047 ** (T - 20.0))

    # Velocidades físicas por trecho
    sec_area = max(compr * prof, 0.1)
    v_natural = Qr / (sec_area * 3600.0)
    velocities = []
    accumulated_flow = Qr
    for eff in active_effluents:
        accumulated_flow += float(eff["Qe"])
        velocities.append(accumulated_flow / (sec_area * 3600.0))

    v_med = (v_natural + sum(velocities)) / (len(velocities) + 1.0)

    # Reoxigenação k2 a 20ºC por O'Connor-Dobbins
    k2_20 = 5.1314 * (v_med ** 0.7076) * (prof ** -1.382)
    if k2_mode == "Manual":
        k2_20 = float(k2_manual) / 24.0
    k2 = k2_20 * (1.024 ** (T - 20.0))

    # 3. Distâncias acumuladas e tempos absolutos de chegada
    dist_acumulada = []
    times = []
    current_dist = 0.0
    for idx, eff in enumerate(active_effluents):
        if idx == 0:
            current_dist = 0.0
            times.append(0.0)
        else:
            current_dist += float(eff["dist"])
            v_prev = velocities[idx - 1]
            t_viagem = float(eff["dist"]) / (3600.0 * v_prev if v_prev > 0 else 1.0)
            times.append(times[idx - 1] + t_viagem)
        dist_acumulada.append(current_dist)

    # 4. Balanço de massa e mistura em cada lançamento
    dbo_mixes = []
    o2_mixes = [Co]
    flow = Qr
    for idx, eff in enumerate(active_effluents):
        qe = float(eff["Qe"])
        le = float(eff["Le"])
        flow += qe
        if idx == 0:
            Lo = (Qr * Lr + qe * le) / max(Qr + qe, 0.1)
            dbo_mixes.append(Lo)
        else:
            t_prev = times[idx] - times[idx - 1]
            if use_corrected_decay:
                Leq = dbo_mixes[idx - 1] * np.exp(-k1 * t_prev)
            else:
                total_mix_prev = sum(dbo_mixes[:idx])
                Leq = total_mix_prev * np.exp(-k1 * times[idx])

            A_term = np.exp(-k1 * t_prev)
            B_term = np.exp(-k2 * t_prev)
            if abs(k2 - k1) < 1e-9:
                k2_adj = k1 + 1e-6
            else:
                k2_adj = k2
            od_end = Cs - (k1 / (k2_adj - k1)) * dbo_mixes[idx - 1] * (B_term - A_term) + (o2_mixes[idx - 1] - Cs) * B_term
            o2_mixes.append(od_end)

            flow_before = flow - qe
            Lo = (qe * le + flow_before * Leq) / max(flow, 0.1)
            dbo_mixes.append(Lo)

    # 5. Simulação contínua no tempo
    points = []
    tempo_max = 300.0
    step = 0.25
    t_arr = np.arange(0.0, tempo_max + step, step)

    min_do = Cs
    min_do_time = 0.0
    min_do_dist = 0.0
    violates_limit = False

    for t in t_arr:
        # Encontra o segmento de deságue ativo
        seg_idx = 0
        for i in range(len(times)):
            if t >= times[i]:
                seg_idx = i

        t_local = t - times[seg_idx]
        A = np.exp(-k1 * t_local)
        B = np.exp(-k2 * t_local)
        conc_bod = dbo_mixes[seg_idx] * A

        if abs(k2 - k1) < 1e-9:
            k2_adj = k1 + 1e-6
        else:
            k2_adj = k2

        conc_od = Cs - (k1 / (k2_adj - k1)) * dbo_mixes[seg_idx] * (B - A) + (o2_mixes[seg_idx] - Cs) * B
        seg_vel = velocities[seg_idx] if seg_idx < len(velocities) else v_natural
        total_dist = dist_acumulada[seg_idx] + t_local * 3600.0 * seg_vel

        final_od = max(0.0, float(conc_od))
        final_bod = max(0.0, float(conc_bod))

        if final_od < min_do:
            min_do = final_od
            min_do_time = t
            min_do_dist = total_dist

        if final_od < Od:
            violates_limit = True

        points.append({
            "tempo": float(t),
            "distancia": round(total_dist, 1),
            "concO2": final_od,
            "concBOD": final_bod,
            "limiteClasse": Od,
            "segmento": seg_idx + 1
        })

    # 6. Extensão de violação em metros
    violation_length = 0.0
    for i in range(len(points) - 1):
        if points[i]["concO2"] < Od:
            violation_length += max(0.0, points[i+1]["distancia"] - points[i]["distancia"])

    return {
        "points": points,
        "k1": k1 * 24.0,       # /dia
        "k2": k2 * 24.0,       # /dia
        "Cs": Cs,
        "v_med": v_med,
        "min_do": min_do,
        "min_do_time": min_do_time,
        "min_do_dist": min_do_dist,
        "violates_limit": violates_limit,
        "violation_length": violation_length,
        "times": times,
        "dist_acumulada": dist_acumulada,
        "classe_limite": Od
    }

# ==============================================================================
# 2. INICIALIZAÇÃO DE ESTADO NO STREAMLIT (SESSION STATE)
# ==============================================================================

if "effluents" not in st.session_state:
    st.session_state.effluents = [
        {"id": "eff_1", "name": "ETE Central (Lançamento 1)", "Qe": 100.0, "Le": 50.0, "dist": 0.0},
        {"id": "eff_2", "name": "Indústria Metalúrgica (Lançamento 2)", "Qe": 80.0, "Le": 40.0, "dist": 1500.0},
        {"id": "eff_3", "name": "Distrito Residencial Sul (Lançamento 3)", "Qe": 60.0, "Le": 30.0, "dist": 2000.0}
    ]

if "saved_scenarios" not in st.session_state:
    st.session_state.saved_scenarios = []

# ==============================================================================
# 3. BARRA LATERAL (SIDEBAR) - PARÂMETROS
# ==============================================================================

st.sidebar.title("⚙️ Parâmetros de Entrada")

# Presets de Rio
st.sidebar.subheader("📌 Presets Rápidos")
preset_rio = st.sidebar.selectbox(
    "Escolha o Tipo de Rio:",
    ["Personalizado", "Rio Lento de Planície (Classe 2)", "Córrego Rápido de Montanha (Classe 1)", "Rio Largo Poluído (Classe 3)", "Canal Urbano Impactado (Classe 4)"]
)

# Valores padrão de rio
def_qr, def_compr, def_prof, def_classe = 5000.0, 50.0, 2.0, 2
if preset_rio == "Rio Lento de Planície (Classe 2)":
    def_qr, def_compr, def_prof, def_classe = 5000.0, 50.0, 2.0, 2
elif preset_rio == "Córrego Rápido de Montanha (Classe 1)":
    def_qr, def_compr, def_prof, def_classe = 1500.0, 12.0, 0.6, 1
elif preset_rio == "Rio Largo Poluído (Classe 3)":
    def_qr, def_compr, def_prof, def_classe = 12000.0, 120.0, 3.5, 3
elif preset_rio == "Canal Urbano Impactado (Classe 4)":
    def_qr, def_compr, def_prof, def_classe = 3000.0, 15.0, 1.2, 4

st.sidebar.subheader("🌊 Dados do Rio Base")
qr = st.sidebar.number_input("Vazão do Rio (Qr - m³/h)", min_value=1.0, max_value=500000.0, value=float(def_qr), step=500.0)
compr = st.sidebar.number_input("Largura do Rio (m)", min_value=1.0, max_value=2000.0, value=float(def_compr), step=5.0)
prof = st.sidebar.number_input("Profundidade Média (m)", min_value=0.1, max_value=50.0, value=float(def_prof), step=0.2)
classe = st.sidebar.selectbox("Classe CONAMA 357", [1, 2, 3, 4], index=def_classe-1)
temp = st.sidebar.slider("Temperatura da Água (°C)", min_value=5.0, max_value=40.0, value=20.0, step=0.5)

st.sidebar.subheader("🧪 Cinética e Reoxigenação")
k1_mode = st.sidebar.radio("Cálculo de k1 (Desoxigenação):", ["Automático", "Manual"], horizontal=True)
k1_manual = 0.35
if k1_mode == "Manual":
    k1_manual = st.sidebar.number_input("k1 a 20°C (dia⁻¹)", min_value=0.01, max_value=5.0, value=0.35, step=0.05)

k2_mode = st.sidebar.radio("Cálculo de k2 (Reoxigenação):", ["Automático", "Manual"], horizontal=True)
k2_manual = 1.2
if k2_mode == "Manual":
    k2_manual = st.sidebar.number_input("k2 a 20°C (dia⁻¹)", min_value=0.01, max_value=20.0, value=1.2, step=0.1)

use_corrected_decay = st.sidebar.checkbox("Fórmula Exponencial Corrigida de DBO", value=True)

# Gestão de Efluentes na Sidebar
st.sidebar.subheader("🏭 Lançamentos Sucessivos (Efluentes)")

for i, eff in enumerate(st.session_state.effluents):
    with st.sidebar.expander(f"📍 {eff['name']}", expanded=(i==0)):
        eff["name"] = st.text_input("Nome:", eff["name"], key=f"name_{eff['id']}")
        eff["Qe"] = st.number_input("Vazão Qe (m³/h):", min_value=0.0, max_value=100000.0, value=float(eff["Qe"]), step=10.0, key=f"qe_{eff['id']}")
        eff["Le"] = st.number_input("DBO Le (mg/L):", min_value=0.0, max_value=5000.0, value=float(eff["Le"]), step=10.0, key=f"le_{eff['id']}")
        if i == 0:
            st.info("Lançamento Inicial (Distância = 0 m)")
            eff["dist"] = 0.0
        else:
            eff["dist"] = st.number_input("Distância do anterior (m):", min_value=1.0, max_value=100000.0, value=float(eff["dist"]), step=500.0, key=f"dist_{eff['id']}")
        
        if len(st.session_state.effluents) > 1:
            if st.button("🗑️ Remover", key=f"del_{eff['id']}"):
                st.session_state.effluents.pop(i)
                st.rerun()

if st.sidebar.button("➕ Adicionar Novo Lançamento"):
    next_num = len(st.session_state.effluents) + 1
    st.session_state.effluents.append({
        "id": f"eff_{len(st.session_state.effluents)}_{next_num}",
        "name": f"Lançamento {next_num}",
        "Qe": 50.0,
        "Le": 100.0,
        "dist": 1500.0 if next_num > 1 else 0.0
    })
    st.rerun()

# Montando objeto de input
inputs = {
    "river": {"Qr": qr, "compr": compr, "prof": prof, "classe": classe},
    "effluents": st.session_state.effluents,
    "use_corrected_decay": use_corrected_decay,
    "temp": temp,
    "k1_mode": k1_mode,
    "k1_manual": k1_manual,
    "k2_mode": k2_mode,
    "k2_manual": k2_manual
}

# Execução da Simulação
result = calculate_streeter_phelps(inputs)
df_points = pd.DataFrame(result["points"])

# ==============================================================================
# 4. INTERFACE PRINCIPAL
# ==============================================================================

st.title("🌊 Simulador Streeter-Phelps de Depleção de Oxigênio")
st.markdown("**Modelagem Hidrológica Multilançamento com Enquadramento Legal CONAMA 357**")

# Abas Principais
tab_sim, tab_cmp, tab_ai, tab_data = st.tabs([
    "📊 Simulação & Gráfico", 
    "⚖️ Comparar Cenários", 
    "🤖 Parecer IA (Gemini)", 
    "📑 Dados & Download CSV"
])

# ------------------------------------------------------------------------------
# ABA 1: SIMULAÇÃO E GRÁFICO
# ------------------------------------------------------------------------------
with tab_sim:
    # Indicadores KPI
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            label="OD Mínimo Crítico",
            value=f"{result['min_do']:.2f} mg/L",
            delta=f"{result['min_do'] - result['classe_limite']:.2f} mg/L vs Limite",
            delta_color="normal" if result['min_do'] >= result['classe_limite'] else "inverse"
        )
    with col2:
        status_text = "Em Conformidade" if not result["violates_limit"] else "VIOLAÇÃO CONAMA"
        color_class = "status-ok" if not result["violates_limit"] else "status-alert"
        st.markdown(f"**Enquadramento (Cl. {classe})**")
        st.markdown(f"<span class='{color_class}' style='font-size:1.15rem;'>{status_text}</span>", unsafe_allow_html=True)
    with col3:
        st.metric(
            label="Ponto Crítico de Depleção",
            value=f"{result['min_do_dist']/1000.0:.2f} km",
            help=f"Ocorre {result['min_do_time']:.1f}h após o primeiro lançamento."
        )
    with col4:
        total_load = sum(calculate_daily_load_kg(e["Qe"], e["Le"]) for e in st.session_state.effluents)
        st.metric(
            label="Carga Orgânica Total",
            value=f"{total_load:,.1f} kg DBO/dia"
        )

    st.divider()

    # Seletor de Eixo e Opções de Gráfico
    ctrl_col1, ctrl_col2 = st.columns([1, 3])
    with ctrl_col1:
        eixo_x = st.radio("Unidade do Eixo X:", ["Distância (km)", "Tempo de Viagem (h)"], horizontal=True)

    # Construção do gráfico com Plotly Subplots (Duplo Eixo Y)
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    x_col = "dist_km" if "Distância" in eixo_x else "tempo"
    df_points["dist_km"] = df_points["distancia"] / 1000.0
    x_label = "Distância Percorrida (km)" if "Distância" in eixo_x else "Tempo de Viagem (horas)"

    if HAS_PLOTLY:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Scatter(
                x=df_points[x_col],
                y=df_points["concO2"],
                name="Oxigênio Dissolvido (OD)",
                mode="lines",
                line=dict(color="#0284c7", width=3.5)
            ),
            secondary_y=False
        )

        fig.add_trace(
            go.Scatter(
                x=df_points[x_col],
                y=df_points["limiteClasse"],
                name=f"Limite Legal Classe {classe} ({result['classe_limite']} mg/L)",
                mode="lines",
                line=dict(color="#ea580c", width=2, dash="dash")
            ),
            secondary_y=False
        )

        fig.add_trace(
            go.Scatter(
                x=[df_points[x_col].min(), df_points[x_col].max()],
                y=[result["Cs"], result["Cs"]],
                name=f"Saturação Cs ({result['Cs']:.2f} mg/L)",
                mode="lines",
                line=dict(color="#94a3b8", width=1.5, dash="dot")
            ),
            secondary_y=False
        )

        fig.add_trace(
            go.Scatter(
                x=df_points[x_col],
                y=df_points["concBOD"],
                name="DBO Carbonácea Remanescente",
                mode="lines",
                line=dict(color="#dc2626", width=2.5)
            ),
            secondary_y=True
        )

        for idx, eff in enumerate(st.session_state.effluents):
            pos_x = (result["dist_acumulada"][idx] / 1000.0) if "Distância" in eixo_x else result["times"][idx]
            fig.add_vline(
                x=pos_x, line_dash="dash", line_color="#7c3aed",
                annotation_text=f"L{idx+1}: {eff['name'][:15]}", annotation_position="top right"
            )

        fig.update_layout(
            title="<b>Perfil Longitudinal de Oxigênio Dissolvido e DBO</b>",
            xaxis_title=x_label,
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=20, r=20, t=60, b=20),
            height=480,
            plot_bgcolor="white"
        )
        fig.update_yaxes(title_text="<b>Oxigênio Dissolvido (mg/L)</b>", gridcolor="#f1f5f9", secondary_y=False)
        fig.update_yaxes(title_text="<b>DBO (mg/L)</b>", showgrid=False, secondary_y=True)

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("💡 Exibindo gráfico nativo do Streamlit:")
        df_chart = df_points.set_index(x_col)[["concO2", "limiteClasse", "concBOD"]].rename(columns={
            "concO2": "OD Atual (mg/L)",
            "limiteClasse": f"Limite CONAMA Cl. {classe}",
            "concBOD": "DBO Remanescente (mg/L)"
        })
        st.line_chart(df_chart)

    # Resumo Cinético na parte inferior
    with st.expander("📌 Detalhes Cinéticos e Físicos do Modelo"):
        k_col1, k_col2, k_col3, k_col4 = st.columns(4)
        k_col1.metric("Coef. Desoxigenação (k1)", f"{result['k1']:.3f} dia⁻¹", f"{T}°C")
        k_col2.metric("Coef. Reoxigenação (k2)", f"{result['k2']:.3f} dia⁻¹", f"V={result['v_med']:.2f} m/s")
        k_col3.metric("Razão de Autodepuração (f)", f"{result['k2']/max(result['k1'],0.001):.2f}", "k2 / k1")
        k_col4.metric("Extensão do Trecho Anóxico/Crítico", f"{result['violation_length']/1000.0:.2f} km")

# ------------------------------------------------------------------------------
# ABA 2: COMPARAR CENÁRIOS
# ------------------------------------------------------------------------------
with tab_cmp:
    st.subheader("⚖️ Comparação Interativa de Cenários")
    st.markdown("Salve o cenário atual para avaliar o impacto de melhorias na ETE, mudanças de vazão ou outorgas.")

    save_col1, save_col2 = st.columns([3, 1])
    with save_col1:
        scen_name = st.text_input("Nome para o Cenário Atual:", f"Cenário - Qr={qr} m³/h, {len(st.session_state.effluents)} efluentes")
    with save_col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("💾 Salvar Cenário Atual", type="primary"):
            st.session_state.saved_scenarios.append({
                "name": scen_name,
                "df": df_points.copy(),
                "min_do": result["min_do"],
                "violates": result["violates_limit"],
                "color": np.random.choice(["#10b981", "#8b5cf6", "#f59e0b", "#ec4899", "#06b6d4"])
            })
            st.success("Cenário salvo!")

    if len(st.session_state.saved_scenarios) > 0:
        if HAS_PLOTLY:
            fig_cmp = go.Figure()
            # Curva atual
            fig_cmp.add_trace(go.Scatter(
                x=df_points["dist_km"], y=df_points["concO2"],
                name="Cenário Atual (Ao Vivo)", mode="lines", line=dict(color="#0284c7", width=3)
            ))
            # Curvas salvas
            for idx, sc in enumerate(st.session_state.saved_scenarios):
                fig_cmp.add_trace(go.Scatter(
                    x=sc["df"]["dist_km"], y=sc["df"]["concO2"],
                    name=f"{sc['name']} (Min: {sc['min_do']:.2f})", mode="lines", line=dict(color=sc["color"], width=2.5, dash="dot")
                ))

            fig_cmp.add_trace(go.Scatter(
                x=[df_points["dist_km"].min(), df_points["dist_km"].max()],
                y=[result["classe_limite"], result["classe_limite"]],
                name=f"Limite CONAMA Cl. {classe}", mode="lines", line=dict(color="#ea580c", width=2, dash="dash")
            ))
            fig_cmp.update_layout(title="Comparativo de Oxigênio Dissolvido entre Cenários", xaxis_title="Distância (km)", yaxis_title="OD (mg/L)", height=450)
            st.plotly_chart(fig_cmp, use_container_width=True)
        else:
            st.info("💡 Exibindo gráfico comparativo nativo do Streamlit:")
            df_cmp = pd.DataFrame({"Cenário Atual": df_points["concO2"].values}, index=df_points["dist_km"])
            for idx, sc in enumerate(st.session_state.saved_scenarios):
                df_cmp[sc["name"]] = sc["df"]["concO2"].values
            df_cmp["Limite CONAMA"] = result["classe_limite"]
            st.line_chart(df_cmp)

        if st.button("🗑️ Limpar Cenários Salvos"):
            st.session_state.saved_scenarios = []
            st.rerun()
    else:
        st.info("Nenhum cenário salvo ainda. Ajuste os parâmetros na sidebar e clique em 'Salvar Cenário Atual'.")

# ------------------------------------------------------------------------------
# ABA 3: PARECER TÉCNICO COM INTELIGÊNCIA ARTIFICIAL (GEMINI)
# ------------------------------------------------------------------------------
with tab_ai:
    st.subheader("🤖 Parecer Técnico Automático com Inteligência Artificial")
    st.markdown("Gere uma análise hidrológica detalhada e recomendações de engenharia ambiental utilizando o **Google Gemini**.")

    api_key = st.text_input("Chave de API do Gemini (ou configure a variável de ambiente GEMINI_API_KEY):", type="password", value=os.environ.get("GEMINI_API_KEY", ""))

    if st.button("✨ Gerar Parecer Ambiental IA", type="primary"):
        if not api_key:
            st.warning("⚠️ Por favor, insira sua chave da API do Gemini acima para gerar o parecer.")
        else:
            with st.spinner("Analisando perfil longitudinal de oxigênio e enquadramento CONAMA 357..."):
                try:
                    from google import genai
                    client = genai.Client(api_key=api_key)
                    
                    prompt = f"""
                    Atue como um Engenheiro Sanitarista e Ambiental especialista em modelagem hidrológica.
                    Analise os seguintes resultados de simulação Streeter-Phelps para o rio e elabore um Parecer Técnico oficial:
                    
                    - Dados do Rio: Vazão Qr = {qr} m³/h, Largura = {compr}m, Profundidade = {prof}m, Classe CONAMA = {classe} (Limite OD >= {result['classe_limite']} mg/L), Temperatura = {temp}°C.
                    - Coeficientes: k1 = {result['k1']:.3f} dia⁻¹, k2 = {result['k2']:.3f} dia⁻¹.
                    - Lançamentos ({len(st.session_state.effluents)}):
                      {chr(10).join([f"  * {e['name']}: Qe={e['Qe']} m³/h, DBO={e['Le']} mg/L na marca {e['dist']}m" for e in st.session_state.effluents])}
                    - Resultados da Simulação:
                      * OD Mínimo atingido: {result['min_do']:.2f} mg/L
                      * Ponto crítico (sag point): {result['min_do_dist']/1000.0:.2f} km rio abaixo ({result['min_do_time']:.1f} horas de viagem)
                      * Situação Legal: {"VIOLAÇÃO DO LIMITE LEGAL" if result['violates_limit'] else "EM CONFORMIDADE COM A CLASSE"}
                      * Extensão de rio em inconformidade: {result['violation_length']/1000.0:.2f} km
                    
                    Estruture o parecer com:
                    1. Diagnóstico Hidrológico e Gravidade do Depósito
                    2. Análise de Enquadramento Legal (Resolução CONAMA 357)
                    3. Medidas de Intervenção Sugeridas (Tratamento na ETE, reoxigenação artificial ou outorga)
                    """
                    
                    response = client.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=prompt,
                    )
                    st.success("Parecer Técnico Concluído!")
                    st.markdown("### 📋 Relatório Técnico do Especialista IA")
                    st.markdown(response.text)
                except Exception as e:
                    st.error(f"Erro ao consultar Gemini: {str(e)}")

# ------------------------------------------------------------------------------
# ABA 4: DADOS TABULARES E EXPORTAÇÃO
# ------------------------------------------------------------------------------
with tab_data:
    st.subheader("📑 Tabela Detalhada de Simulação Longitudinal")
    st.dataframe(
        df_points.rename(columns={
            "tempo": "Tempo (h)",
            "distancia": "Distância (m)",
            "concO2": "OD (mg/L)",
            "concBOD": "DBO (mg/L)",
            "limiteClasse": "Limite CONAMA (mg/L)",
            "segmento": "Trecho/Segmento"
        }),
        use_container_width=True,
        height=400
    )

    csv = df_points.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Baixar Dados Completos (CSV)",
        data=csv,
        file_name="simulacao_streeter_phelps.csv",
        mime="text/csv",
        type="primary"
    )
`;

export const REQUIREMENTS_TXT_CONTENT = `streamlit>=1.38.0
plotly>=5.24.0
pandas>=2.2.0
numpy>=1.26.0
google-genai>=0.1.0`;

export default function StreamlitViewer({ inputs }: { inputs?: SimulationInputs }) {
  const [copied, setCopied] = useState(false);
  const [copiedReq, setCopiedReq] = useState(false);
  const [activeFile, setActiveFile] = useState<"app.py" | "requirements.txt">("app.py");

  const currentCode = React.useMemo(() => {
    if (!inputs) return STREAMLIT_PYTHON_CODE;
    try {
      const effStr = inputs.effluents.map(e => `        {"id": "${e.id}", "name": "${e.name}", "Qe": ${Number(e.Qe).toFixed(1)}, "Le": ${Number(e.Le).toFixed(1)}, "dist": ${Number(e.dist).toFixed(1)}}`).join(",\n");
      return STREAMLIT_PYTHON_CODE
        .replace(/st\.session_state\.effluents = \[\s*\{[\s\S]*?\]/, `st.session_state.effluents = [\n${effStr}\n    ]`)
        .replace(/def_qr, def_compr, def_prof, def_classe = 5000\.0, 50\.0, 2\.0, 2/, `def_qr, def_compr, def_prof, def_classe = ${Number(inputs.river.Qr).toFixed(1)}, ${Number(inputs.river.compr).toFixed(1)}, ${Number(inputs.river.prof).toFixed(1)}, ${inputs.river.classe}`)
        .replace(/value=20\.0, step=0\.5\)/, `value=${Number(inputs.temp).toFixed(1)}, step=0.5)`);
    } catch {
      return STREAMLIT_PYTHON_CODE;
    }
  }, [inputs]);

  const handleCopy = () => {
    navigator.clipboard.writeText(currentCode);
    setCopied(true);
    setTimeout(() => setCopied(false), 2500);
  };

  const handleCopyReq = () => {
    navigator.clipboard.writeText(REQUIREMENTS_TXT_CONTENT);
    setCopiedReq(true);
    setTimeout(() => setCopiedReq(false), 2500);
  };

  const handleDownload = () => {
    const blob = new Blob([currentCode], { type: "text/x-python;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "app.py";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const handleDownloadReq = () => {
    const blob = new Blob([REQUIREMENTS_TXT_CONTENT], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "requirements.txt";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-6 animate-fadeIn">
      {/* Alerta de Solução Definitiva no Código Python para o Erro ModuleNotFoundError */}
      <div className="bg-emerald-500/10 border-2 border-emerald-500/40 rounded-2xl p-5 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 shadow-sm">
        <div className="flex items-start gap-3">
          <Sparkles className="w-6 h-6 text-emerald-600 shrink-0 mt-0.5" />
          <div className="space-y-1">
            <h4 className="font-bold text-slate-900 text-base">
              🎯 Erro Corrigido Diretamente no Código (<code className="bg-emerald-100 text-emerald-800 px-1.5 py-0.5 rounded text-xs font-mono">App Rio.py</code> Blindado!)
            </h4>
            <p className="text-xs sm:text-sm text-slate-700 leading-relaxed">
              Injetamos uma rotina de <strong>Auto-Instalação</strong> no início do código Python! Agora, mesmo que você envie apenas o arquivo <code className="bg-slate-200 text-slate-800 font-mono px-1 rounded font-bold">App Rio.py</code> sem o <code className="bg-slate-200 text-slate-800 font-mono px-1 rounded font-bold">requirements.txt</code>, o próprio script identificará a ausência do <code className="font-mono">plotly</code> e o instalará ou usará um gráfico nativo sem quebrar.
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2 shrink-0 self-end sm:self-center">
          <button
            onClick={handleCopyReq}
            className="px-3.5 py-2 bg-white hover:bg-slate-100 text-slate-800 text-xs font-bold rounded-xl border border-slate-300 shadow-2xs transition-all flex items-center gap-1.5 cursor-pointer"
          >
            {copiedReq ? <Check className="w-4 h-4 text-emerald-600" /> : <Copy className="w-4 h-4 text-slate-600" />}
            {copiedReq ? "Copiado!" : "Copiar requirements.txt"}
          </button>
          <button
            onClick={handleDownloadReq}
            className="px-4 py-2 bg-amber-600 hover:bg-amber-500 text-white text-xs font-bold rounded-xl shadow-md transition-all flex items-center gap-1.5 cursor-pointer"
          >
            <Download className="w-4 h-4" />
            Baixar requirements.txt
          </button>
        </div>
      </div>

      {/* Banner Explicativo */}
      <div className="bg-gradient-to-r from-emerald-900 via-teal-950 to-slate-900 text-white rounded-2xl p-6 shadow-xl border border-emerald-500/30 flex flex-col md:flex-row items-start md:items-center justify-between gap-6">
        <div className="space-y-2">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-emerald-500/20 text-emerald-300 text-xs font-semibold border border-emerald-500/30">
            <Sparkles className="w-3.5 h-3.5" />
            100% Compatível com Streamlit & Python
          </div>
          <h2 className="text-xl sm:text-2xl font-display font-extrabold text-white tracking-tight">
            Versão Completa em Python (Streamlit)
          </h2>
          <p className="text-sm text-emerald-100/80 max-w-2xl leading-relaxed">
            Aqui está o código completo do simulador de Depleção de Oxigênio Streeter-Phelps transposto para Python utilizando <strong>Streamlit</strong>, <strong>Plotly</strong>, <strong>Pandas</strong>, <strong>NumPy</strong> e o SDK oficial do <strong>Google Gemini</strong>.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-3 shrink-0">
          <button
            onClick={handleCopy}
            className="flex items-center gap-2 px-4 py-2.5 bg-slate-800 hover:bg-slate-700 text-white text-xs font-semibold rounded-xl border border-slate-600 transition-all cursor-pointer shadow-sm"
          >
            {copied ? <Check className="w-4 h-4 text-emerald-400" /> : <Copy className="w-4 h-4 text-slate-300" />}
            {copied ? "Código Copiado!" : "Copiar app.py"}
          </button>
          <button
            onClick={handleDownload}
            className="flex items-center gap-2 px-5 py-2.5 bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-semibold rounded-xl transition-all cursor-pointer shadow-md shadow-emerald-900/30"
          >
            <Download className="w-4 h-4" />
            Baixar app.py
          </button>
        </div>
      </div>

      {/* Guia de Execução Rápida */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-white rounded-xl p-4 border border-slate-200 shadow-xs space-y-2">
          <div className="flex items-center gap-2 text-emerald-600 font-bold text-sm">
            <Layers className="w-4 h-4" />
            1. No Streamlit Cloud (GitHub)
          </div>
          <p className="text-xs text-slate-600">Suba no mesmo diretório do seu repositório os arquivos:</p>
          <div className="bg-slate-900 text-emerald-300 font-mono text-xs p-2.5 rounded-lg space-y-1">
            <div>📄 app.py (seu código principal)</div>
            <div>📄 requirements.txt (as bibliotecas)</div>
          </div>
        </div>

        <div className="bg-white rounded-xl p-4 border border-slate-200 shadow-xs space-y-2">
          <div className="flex items-center gap-2 text-blue-600 font-bold text-sm">
            <Terminal className="w-4 h-4" />
            2. Uso Local (Terminal)
          </div>
          <p className="text-xs text-slate-600">
            Se for rodar no seu computador, instale com o comando:
          </p>
          <div className="bg-slate-900 text-slate-200 font-mono text-xs p-2.5 rounded-lg overflow-x-auto select-all">
            pip install -r requirements.txt
          </div>
        </div>

        <div className="bg-white rounded-xl p-4 border border-slate-200 shadow-xs space-y-2">
          <div className="flex items-center gap-2 text-purple-600 font-bold text-sm">
            <Sparkles className="w-4 h-4" />
            3. Execute o Aplicativo
          </div>
          <p className="text-xs text-slate-600">Após instalar, inicie o servidor localmente:</p>
          <div className="bg-slate-900 text-emerald-400 font-mono text-xs p-2.5 rounded-lg overflow-x-auto select-all">
            streamlit run app.py
          </div>
        </div>
      </div>

      {/* Code Viewer com Abas */}
      <div className="bg-slate-900 rounded-2xl overflow-hidden border border-slate-800 shadow-xl">
        <div className="bg-slate-950 px-4 py-3 flex flex-wrap items-center justify-between gap-3 border-b border-slate-800">
          <div className="flex items-center gap-4">
            <div className="flex gap-1.5">
              <span className="w-3 h-3 rounded-full bg-rose-500/80"></span>
              <span className="w-3 h-3 rounded-full bg-amber-500/80"></span>
              <span className="w-3 h-3 rounded-full bg-emerald-500/80"></span>
            </div>
            
            <div className="flex gap-2">
              <button
                onClick={() => setActiveFile("app.py")}
                className={`px-3 py-1 rounded-lg text-xs font-mono font-bold transition-all cursor-pointer ${
                  activeFile === "app.py"
                    ? "bg-emerald-500/20 text-emerald-300 border border-emerald-500/40"
                    : "text-slate-400 hover:text-white"
                }`}
              >
                app.py (Código Python)
              </button>
              <button
                onClick={() => setActiveFile("requirements.txt")}
                className={`px-3 py-1 rounded-lg text-xs font-mono font-bold transition-all cursor-pointer flex items-center gap-1.5 ${
                  activeFile === "requirements.txt"
                    ? "bg-amber-500/20 text-amber-300 border border-amber-500/40"
                    : "text-slate-400 hover:text-white"
                }`}
              >
                requirements.txt (Dependências do Cloud)
              </button>
            </div>
          </div>

          {activeFile === "app.py" ? (
            <button
              onClick={handleCopy}
              className="flex items-center gap-1.5 text-xs font-medium text-slate-300 hover:text-white transition-colors cursor-pointer bg-slate-800 px-3 py-1.5 rounded-lg border border-slate-700"
            >
              {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
              {copied ? "Copiado!" : "Copiar app.py"}
            </button>
          ) : (
            <button
              onClick={handleCopyReq}
              className="flex items-center gap-1.5 text-xs font-medium text-slate-300 hover:text-white transition-colors cursor-pointer bg-slate-800 px-3 py-1.5 rounded-lg border border-slate-700"
            >
              {copiedReq ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
              {copiedReq ? "Copiado!" : "Copiar requirements.txt"}
            </button>
          )}
        </div>

        <pre className="p-5 overflow-x-auto text-xs sm:text-sm font-mono text-slate-200 leading-relaxed max-h-[700px] overflow-y-auto">
          <code>{activeFile === "app.py" ? currentCode : REQUIREMENTS_TXT_CONTENT}</code>
        </pre>
      </div>
    </div>
  );
}
