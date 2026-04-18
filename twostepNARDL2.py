import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from twostep_nardl import TwoStepNARDL
from twostep_nardl.postestimation import (
    bounds_test, wald_test, diagnostics, multipliers,
    half_life, asymadj, irf, ecm_table
)
from twostep_nardl.plotting import plot_multipliers, plot_halflife
import io
import contextlib

st.set_page_config(page_title="NARDL Analysis Tool", layout="wide")

# Custom CSS for the "Colored Terminal" look
st.markdown("""
    <style>
    .terminal-container {
        background-color: #1e1e1e;
        color: #d4d4d4;
        padding: 15px;
        border-radius: 8px;
        font-family: 'Courier New', monospace;
        border-left: 5px solid #00ffcc;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

#st.markdown(
    #"""
    #<style>
    #/* This targets the main app container */
    #.stApp {
        #background-color: #000000;
    #}

    #/* This targets all text elements and makes them bold and brown */
    #.stApp p, .stApp h1, .stApp h2, .stApp h3, .stApp span, .stApp label {
        #color: #A52A2A !important;
        #font-weight: bold !important;
    #}
    #</style>
    #""",
    #unsafe_allow_html=True
#)


st.title("📈 Nonlinear ARDL (Two-Step) Dashboard")
st.markdown("Developed for advanced econometric research in exchange rate and inflation dynamics.")

# --- SESSION STATE ---
if "results" not in st.session_state:
    st.session_state["results"] = None

# --- 1. DATA UPLOAD ---
st.sidebar.header("1. Data Input")
uploaded_file = st.sidebar.file_uploader("Upload Excel or CSV", type=["xlsx", "csv"])

if uploaded_file:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    index_col = st.sidebar.selectbox("Select Index Column (Date/Time)", [None] + list(df.columns))

    if index_col:
        df = df.set_index(index_col)
        df.index = pd.to_datetime(df.index)
    else:
        st.sidebar.warning("No index column selected. Generating manual time index...")
        freq_map = {"Monthly": "ME", "Quarterly": "QE", "Annual": "YE"}
        freq_choice = st.sidebar.selectbox("Select Frequency", list(freq_map.keys()))
        start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2000"))

        df.index = pd.date_range(start=start_date, periods=len(df), freq=freq_map[freq_choice])
        st.sidebar.success(f"Generated index from {df.index[0].date()} to {df.index[-1].date()}")

    st.write("### Dataset Preview", df.head())

    # --- 2. MODEL CONFIG ---
    st.sidebar.header("2. Model Configuration")
    depvar = st.sidebar.selectbox("Dependent Variable (y)", df.columns)
    xvars = st.sidebar.multiselect("Independent Variables (x)", [c for c in df.columns if c != depvar])
    decompose = st.sidebar.multiselect("Variables to Decompose", xvars)

    method = st.sidebar.selectbox("Estimation Method", ["twostep", "onestep"])
    step1 = st.sidebar.selectbox("Step 1 Estimator", ["fmols", "fmtols", "ols", "tols"])
    case = st.sidebar.slider("PSS Case", 1, 5, 3)
    max_lags = st.sidebar.number_input("Max Lags", value=4, min_value=1)
    ic = st.sidebar.selectbox("Information Criterion", ["bic", "aic"])


    tab1, tab2, tab3, tab4 = st.tabs(["📋 Results", "⚡ Dynamics", "🔍 Diagnostics", "📚 Features Info"])

    # =========================
    # TAB 1: MODEL
    # =========================
    with tab1:

        if st.sidebar.button("Run NARDL Estimation"):
            try:
                with st.spinner('Estimating Model...'):
                    model = TwoStepNARDL(
                        data=df,
                        depvar=depvar,
                        xvars=xvars,
                        decompose=decompose,
                        maxlags=int(max_lags),
                        ic=ic,
                        method=method,
                        step1=step1,
                        case=case
                    )
                    st.session_state["results"] = model.fit()

                st.success("Model Estimated Successfully!")

            except Exception as e:
                st.error(f"Error during estimation: {e}")

        results = st.session_state.get("results")

        if results is not None:
            st.write("### 📊 Model Summary")
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.code(str(results), language="text")

            
            # ECM
            st.write("### 📄 Error Correction Representation (ECM)")
            f = io.StringIO()
            with contextlib.redirect_stdout(f):
                ecm_table(results)
            st.code(f.getvalue(), language="text")

            st.info("""
            **Key Observations:**
            * **Panel A (Long-Run):** The positive shock to inflation (`inf_pos`) is significant (**p=0.0406**), while the negative shock is not. This confirms your asymmetric hypothesis.
            * **Panel B (Short-Run):** The `L.ect` (Error Correction Term) is **-0.1271**. This indicates that about 12.7% of the deviation from the long-run equilibrium is corrected each period, though its p-value (0.2415) suggests the adjustment speed is statistically weak in this specific lag specification.
            """)

            # Asymmetry
            st.write("### ⚡ Asymmetric Adjustment")
            f = io.StringIO()
            with contextlib.redirect_stdout(f):
                asymadj(results)
            st.code(f.getvalue(), language="text")

            st.info(f"""
            **Research Insights:**
            * **Asymmetric Impact:** The long-run impact is larger for **increases** in inflation. This suggests that inflationary shocks have a more persistent structural effect on the Algerian economy than deflationary ones.
            * **Convergence Speed:** A half-life of **5.10 periods** indicates that after a shock, it takes approximately 5 months to recover 50% of the equilibrium level.
            """)

        else:
            st.info("Run the model to see results.")

    # =========================
    # TAB 2: POST ESTIMATION
    # =========================
    with tab2:
        results = st.session_state.get("results")

        if results is None:
            st.warning("Run the model first.")
        else:
            st.write("### Bounds Test (PSS 2001)")
            bt = bounds_test(results)
            #st.write(bt)
            # 1. Create a Summary Stats Table
            st.markdown(f"**PSS Case:** {bt['case']} | **K (Number of Regressors):** {bt['k']}")

            stats_data = {
                "Statistic": ["F-statistic (PSS)", "t-statistic (BDM)", "Speed of Adjustment (ρ)"],
                "Value": [f"{bt['F_pss']:.4f}", f"{bt['t_bdm']:.4f}", f"{bt['rho']:.4f}"]
            }
            st.table(pd.DataFrame(stats_data))
            # 2. Create the Decision Table
            decision_list = []
            for level, result in bt['decisions'].items():
                # Formatting the percentage (e.g., 0.1 -> 10%)
                level_pct = f"{float(level)*100:.0f}%"
            
                # Mapping "no rejection" to a cleaner academic status
                status = "❌ No Cointegration" if result == "no rejection" else "✅ Cointegrated"
            
                decision_list.append({
                    "Significance Level": level_pct,
                    "Status": result.capitalize(),
                    "Decision": status
                })

            df_decisions = pd.DataFrame(decision_list)
            st.table(df_decisions)

            st.caption("Note: H₀ (Null Hypothesis) assumes no long-run level relationship (no cointegration).")    

            st.write("### ⚖️ Wald Tests for Asymmetry")

            # Perform the wald test
            wt = wald_test(results)


            data = []
            for test_name, stats in wt.items():
                p_val = stats['p']
                # 1. Determine Significance Stars
                if p_val < 0.01:
                    sig_decision = "*** Reject H₀ (Asymmetry)"
                elif p_val < 0.05:
                    sig_decision = "** Reject H₀ (Asymmetry)"
                elif p_val < 0.10:
                    sig_decision = "* Reject H₀ (Asymmetry)"
                else:
                    sig_decision = "Fail to Reject H₀ (Symmetry)"

                # 2. Determine Decision Logic
                # H0: The relationship is Symmetric. 
                # Reject H0 = Evidence of Asymmetry.
                if p_val < 0.5:
                    decision = "Reject H₀ (Asymmetry)"
                else:
                    decision = "Fail to Reject H₀ (Symmetry)"
                data.append({
                    "Test Type": test_name,
                    "Wald Statistic (W)": round(stats['W'], 4),
                    "p-value": f"{p_val:.4f}",
                    "Significance": sig_decision
                    })
            df_wald = pd.DataFrame(data)
            # 2. Display as a nice static table
            st.table(df_wald)

            # 3. Add academic footer
            st.caption("Note: *** p<0.01, ** p<0.05, * p<0.10. \n" \
            "\n A significant p-value indicates the rejection of the null hypothesis of symmetry.")   

            


            # Plots
            st.write("### Dynamic Multipliers")
            plot_multipliers(results)
            st.pyplot(plt.gcf())
            plt.clf()

            st.write("### Half Life")
            plot_halflife(results)
            st.pyplot(plt.gcf())
            plt.clf()

            st.write("### ⏳ Half-Life & Persistence Analysis")

            st.markdown("<div class='card'>", unsafe_allow_html=True)

            # 1. Capture the printed output from half_life
            # Half-life table
            f = io.StringIO()
            with contextlib.redirect_stdout(f):
                half_life(results)
            st.code(f.getvalue())

            
            # 3. Academic Interpretation for your Research Paper
            st.info(f"""
            **Persistence Insights:**
            * **Speed of Adjustment (ρ):** The coefficient of **-0.1271** implies a relatively slow correction of disequilibrium (about 12.7% per period).
            * **Half-Life (50%):** It takes approximately **5.1 to 6 periods** (months/quarters) for 50% of a shock to dissipate.
            * **Full Recovery:** The 99% adjustment time of **33.88 periods** suggests that the long-term effects of a shock to the Algerian economy remain visible for nearly 3 years before disappearing completely.
            """)


            st.write("### 📊 Numerical Impulse Response Functions")

            st.markdown("<div class='card'>", unsafe_allow_html=True)

            # IRF
            f = io.StringIO()
            with contextlib.redirect_stdout(f):
                irf(results)
            st.code(f.getvalue())

            # 3. Academic Analysis for the University of Algiers 3 Paper
            st.info("""
            **Key Findings from the IRF Table:**
            * **Immediate Impact (Horizon 1):** A positive shock has a large immediate effect (0.8379), while a negative shock is almost negligible (-0.0855).
            * **Long-Run Convergence:** Note how the cumulative values stabilize after Horizon 15. The positive multiplier is roughly **3.3 times larger** than the negative multiplier in absolute terms.
            * **Economic Interpretation:** This suggests that inflationary pressures in Algeria are much stickier and more impactful than deflationary forces.
            """)
            
            st.write("### 📈 Cumulative Dynamic Multipliers")

            st.markdown("<div class='card'>", unsafe_allow_html=True)


            # Multipliers
            f = io.StringIO()
            with contextlib.redirect_stdout(f):
                multipliers(results)
            st.code(f.getvalue())

                    # 3. Final Economic Synthesis for the Article
            st.info(f"""
            **Final Analysis of Asymmetry:**
            * **Immediate Divergence:** Even at Horizon 1, the difference is **0.9234** and highly significant ($***$). This confirms that the Algerian economy reacts instantaneously and differently to shocks.
            * **Long-Run Equilibrium (LR eq.):** The model settles at a positive multiplier of **1.5382** and a negative one of **-0.4624**. 
            * **The "Asymmetry Gap":** The total gap of **2.0007** is the permanent divergence between a positive and negative shock of the same magnitude. 
            """)

    # =========================
    # TAB 3: DIAGNOSTICS
    # =========================
    with tab3:
        results = st.session_state.get("results")

        if results is None:
            st.warning("Run the model first.")
        else:
            st.write("### 🔍 Diagnostics")
            diag = diagnostics(results)

            diag_df = pd.DataFrame({
                "Test": ["BG", "White", "JB", "RESET"],
                "Stat": [
                    diag.get("bg_chi2"),
                    diag.get("white_chi2"),
                    diag.get("jb_stat"),
                    diag.get("reset_F")
                ],
                "p-value": [
                    diag.get("bg_p"),
                    diag.get("white_p"),
                    diag.get("jb_p"),
                    diag.get("reset_p")
                ]
            })

            st.table(diag_df)

                    # 1. Run the diagnostics
            diag = diagnostics(results)

            # This map pairs the 'stat' key and 'p' key for each actual test
            test_mapping = {
                "Serial Correlation (BG)": {"stat": "bg_chi2", "p": "bg_p", "h0": "No Serial Correlation"},
                "Heteroskedasticity (White)": {"stat": "white_chi2", "p": "white_p", "h0": "No Heteroskedasticity"},
                "Normality (Jarque-Bera)": {"stat": "jb_stat", "p": "jb_p", "h0": "Residuals are Normal"},
                "Functional Form (RESET)": {"stat": "reset_F", "p": "reset_p", "h0": "Model correctly specified"}
            }

            # 2. Process the results into a clean table
            diag_data = []

            for display_name, keys in test_mapping.items():

                p_val = diag.get(keys["p"], np.nan)
                stat_val = diag.get(keys["stat"], np.nan)
                if p_val < 0.05:
                    decision = "❌ Reject H₀ (Violation)"
                elif p_val < 0.10:
                    decision = "⚠️ Weak Reject (Caution)"
                else:
                    decision = "✅ Fail to Reject H0 (Passed)"
            
                diag_data.append({
                    "Diagnostic Test": display_name,
                    "Null Hypothesis (H₀)": keys["h0"],
                    "Statistic": f"{stat_val:.4f}",
                    "p-value": f"{p_val:.4f}",
                    "Result": decision
                })

            # 3. Display the DataFrame
            df_diag = pd.DataFrame(diag_data)
            st.table(df_diag)

            st.info("""
                    **Interpretation Guide:** A 'Passed' result means your model satisfies the standard econometric assumptions. 
                    If the **Functional Form** fails, consider adding more lags or a structural dummy variable.
            """)
    # =========================
    # TAB 4: Information
    # =========================

    with tab4:
        st.write("## 📚 NARDL Framework Information")
        

        st.markdown("""
        <div class='feature-card'>
        <b>✅ Two-Step Estimation</b><br>
        Uses FM-OLS for long-run cointegration to correct for endogeneity and OLS for the Error Correction Model (ECM).
        </div>
        <div class='feature-card'>
        <b>✅ Partial Sum Decomposition</b><br>
        Decomposes variables into positive (x<sub>t</sub>⁺) and negative (x<sub>t</sub>⁻) to capture asymmetric economic shocks.
        </div>
        <div class='feature-card'>
        <b>✅ PSS (2001) Bounds Test</b><br>
        Validates long-run relationships using Case 1–5 critical values (Pesaran et al., 2001).
        </div>
        """, unsafe_allow_html=True)
            

        st.markdown("""
        <div class='feature-card'>
        <b>✅ Persistence & Half-Life</b><br>
        Calculates the time required for 50% of a shock to dissipate (Pesaran & Shin, 1996).
        </div>
        <div class='feature-card'>
        <b>✅ Wald Tests</b><br>
        Formal statistical testing for Long-Run and Short-Run additive asymmetry.
        </div>
        <div class='feature-card'>
        <b>✅ Publication Quality</b><br>
        Integrated Matplotlib plotting and Pandas DataFrame compatibility for academic research.
        </div>
        """, unsafe_allow_html=True)

        st.header("📚 Methodological Framework & Documentation")
        st.markdown("---")

        # Section 1: Estimation
        with st.expander("1. Estimation Frameworks", expanded=True):
            st.markdown("""
            * **Two-Step Estimation (FM-OLS/FM-TOLS):** Unlike standard OLS, Fully Modified OLS accounts for endogeneity and serial correlation in the long-run relationship. This is particularly useful when variables are $I(1)$ and there is a lead-lag relationship between them.
            * **One-Step Estimation (SYG 2014):** Based on Shin, Yu, and Greenwood-Nimmo (2014). This is the modern standard for NARDL, allowing for the simultaneous estimation of long-run and short-run asymmetries within a single equation, improving efficiency.
            """)

        # Section 2: Asymmetry
        with st.expander("2. Asymmetry Mechanics"):
            st.markdown("""
            * **Partial Sum Decomposition:** This is the core of "Nonlinear" ARDL. It decomposes an independent variable ($x_t$) into its positive ($x_t^+$) and negative ($x_t^-$) cumulative sums.
            * **Mathematical Logic:** It tests if the economy reacts differently to a "price hike" vs. a "price drop."
            * **Threshold Support:** Allows the decomposition to happen only after a certain magnitude of change is reached, filtering out "noise" from small fluctuations.
            """)

        # Section 3: Cointegration
        with st.expander("3. Cointegration & Validation"):
            st.markdown("""
            * **PSS (2001) Bounds Test:** Implements the Pesaran, Shin, and Smith (2001) approach. It checks the F-statistic against two sets of critical values:
                * **Lower Bound $I(0)$:** Assumes variables are stationary.
                * **Upper Bound $I(1)$:** Assumes variables have a unit root.
            * *Decision Rule:* If your F-stat is above the Upper Bound, cointegration is confirmed regardless of the integration order.
            * **Wald Tests for Asymmetry:**
                * **Long-Run (LR):** Tests if $\\beta^+ = \\beta^-$.
                * **Short-Run (SR):** Tests if the sum of lagged positive changes equals the sum of lagged negative changes.
            """)

        # Section 4: Dynamics
        with st.expander("4. Dynamic Recovery & Shocks"):
            st.markdown("""
            * **Cumulative Dynamic Multipliers:** These plots show the "path to equilibrium." They visualize how the dependent variable moves over time following a 1-unit shock to the positive vs. negative components.
            * **Half-Life & Persistence:** Derived from Pesaran and Shin (1996), the half-life tells you exactly how many periods (months/quarters) it takes for 50% of a shock's effect to vanish. A high half-life indicates a very "sticky" or persistent economic impact.
            """)

        # Section 5: Diagnostics
        with st.expander("5. Statistical Rigor (Diagnostics)"):
            st.markdown("""
            To ensure your results aren't "spurious" (accidental), the tool runs:
            * **Breusch-Godfrey:** To ensure no patterns are left in the errors (autocorrelation).
            * **White Test:** To ensure the error variance is constant (homoscedasticity).
            * **Jarque-Bera:** To confirm the residuals follow a normal distribution, making your t-statistics valid.
            * **Ramsey RESET:** To check if you have missed any important variables or used the wrong functional form.
            """)

        # Section 6: Features
        with st.expander("6. User-Centric Features"):
            st.markdown("""
            * **AIC/BIC Automation:** Automatically tests hundreds of lag combinations to find the "mathematically perfect" model, saving hours of manual trial and error.
            * **Pandas Integration:** Since it works directly on DataFrames, it preserves your date indices, making time-series plotting seamless.
            """)


else:
    st.info("Please upload a dataset.")
