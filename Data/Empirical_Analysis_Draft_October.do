/*
Expansionary Fiscal Consolidation Under Sovereign Risk

This do-file conduct the empirical analysis presented in Section 4 of the final draft - October 2024
Specifically, Table 5-7 and Figure 5, 9-12.

Authors: Carlos Esquivel and Agustin Samano
*/
		
	*Agustin's directory
		global input = "C:\Users\WB581020\OneDrive - WBG\1.Independent Research\CarlosAgustin\Expansionary Fiscal Consolidation Under Sovereign Risk\Empirical Analysis\Inputs"
        global output = "C:\Users\WB581020\OneDrive - WBG\1.Independent Research\CarlosAgustin\Expansionary Fiscal Consolidation Under Sovereign Risk\Empirical Analysis\Results"
        global programs = "C:\Users\WB581020\OneDrive - WBG\1.Independent Research\CarlosAgustin\Expansionary Fiscal Consolidation Under Sovereign Risk\Empirical Analysis\DoFiles"
	
************************************************************************************
* Table 5-7. BASELINE: PANEL REGRESSION using private investment as % of GDP as dependent variable
************************************************************************************	

	* Load the main database
	    use "$input/fiscal_rules_macro_variables_2000_2019_readytouse.dta", replace
	* Preserve the dataset to restore it later	
		preserve
	* Keep only countries with enough data
		drop if ipriv_gdp == .
		drop if spreads == .
		drop if ggdy == .
		drop if gdp_cc == .	
	* Eliminate countries with less than 5 observations
		bysort country_code: gen obs_count = _N
		drop if obs_count < 5
		drop obs_count  // Optional: drop the temporary obs_count variable if no longer needed
	* Estimating Panel Regressions
		xtreg ipriv_gdp l.drdebtrule, robust
		outreg2 using "$output\Table_5.xls", replace ctitle(Baseline)
		xtreg ipriv_gdp l.drdebtrule l.spreads, robust
		outreg2 using "$output\Table_5.xls", append ctitle(Sovereign Spreads)
		xtreg ipriv_gdp l.drdebtrule l.spreads l.ggdy, robust
		outreg2 using "$output\Table_5.xls", append ctitle(Public Debt)	
		xtreg ipriv_gdp l.drdebtrule l.spreads l.ggdy l.gdp_cc, robust
		outreg2 using "$output\Table_5.xls", append ctitle(GDP)	
	* Estimating Panel Regressions + country FE
		xtreg ipriv_gdp l.drdebtrule, fe robust
		outreg2 using "$output\Table_6.xls", replace ctitle(Baseline)
		xtreg ipriv_gdp l.drdebtrule l.spreads, fe robust
		outreg2 using "$output\Table_6.xls", append ctitle(Sovereign Spreads)
		xtreg ipriv_gdp l.drdebtrule l.spreads l.ggdy, fe robust
		outreg2 using "$output\Table_6.xls", append ctitle(Public Debt)	
		xtreg ipriv_gdp l.drdebtrule l.spreads l.ggdy l.gdp_cc, fe robust
		outreg2 using "$output\Table_6.xls", append ctitle(GDP)		
	* Gen log vars
		gen ln_ipriv   = log(ipriv_gdp)
		gen ln_spreads = log(spreads)
		gen ln_ggdy    = log(ggdy)
	* Estimating Panel Regressions with logs
		xtreg ln_ipriv l.drdebtrule, robust
		outreg2 using "$output\Table_7.xls", replace ctitle(Baseline)
		xtreg ln_ipriv l.drdebtrule l.ln_spreads, robust
		outreg2 using "$output\Table_7.xls", append ctitle(Sovereign Spreads)
		xtreg ln_ipriv l.drdebtrule l.ln_spreads l.ln_ggdy, robust
		outreg2 using "$output\Table_7.xls", append ctitle(Public Debt)	
		xtreg ln_ipriv l.drdebtrule l.ln_spreads l.ln_ggdy l.gdp_cc, robust
		outreg2 using "$output\Table_7.xls", append ctitle(GDP)	
	* Export Excel
		keep country year ipriv_gdp drdebtrule spreads ggdy gdp_cc
		export excel using "$output/panel_reg_sample_baseline.xlsx", replace firstrow(variables)	
	* Restore the original dataset
		restore
		
************************************************************************************
* Figure 5. BASELINE: LOCAL PROJECTIONS using private investment as dependent variable
************************************************************************************	
	* Load the main database
	    use "$input/fiscal_rules_macro_variables_2000_2019_readytouse.dta", replace
	* Preserve the dataset to restore it later	
		preserve	
	* Keep only countries with enough data
		drop if ipriv_gdp == .
		drop if spreads == .
		drop if ggdy == .
		drop if gdp_cc == .	
	* Eliminate countries with less than 5 observations
		bysort country_code: gen obs_count = _N
		drop if obs_count < 5
		drop obs_count  // Optional: drop the temporary obs_count variable if no longer needed		
	* Defining fiscal shock as a dummy variable for debt rule adoption
		gen dr_adoption = 0
		replace dr_adoption = 1 if drdebtrule == 1 & l.drdebtrule == 0	
	* Mark the sample before running lpirf to flag the exact sample used
		mark sample_flag	
	* Estimating Local Projections (LPIRF)
		lpirf ipriv_gdp, lags(1/2) step(6) exog(dr_adoption l.spreads l.ggdy l.gdp_cc) dfk small vce(robust) level(90)
		outreg2 using "$output\LPIRF.xls", replace 
	* Save the IRF results, replacing if necessary	
		irf set lpirf.irf, replace
	* Create IRF results using the label "IRF"
		irf create IRF
	*Plot the IRF - Private Investment
		irf graph dm, irf(IRF) impulse(dr_adoption) response(ipriv_gdp) ///
		yline(0, lcolor(black) lpattern(dash)) xlabel(0(1)5) ///
		ylabel(-4(1)4) xtitle("Time Horizon (Years)") ///
		ytitle("% GDP") ///
		legend(off)	level(90)
	* Export the exact sample used in lpirf to Excel
		keep if sample_flag == 1
		export excel using "$output/lpirf_exact_sample_baseline.xlsx", replace firstrow(variables)
	* Restore the original dataset
		restore
	
************************************************************************************
* Figure 9. LOCAL PROJECTIONS using private investment as dependent variable (4 STEPS)
************************************************************************************	
	* Load the main database
	    use "$input/fiscal_rules_macro_variables_2000_2019_readytouse.dta", replace
	* Preserve the dataset to restore it later	
		preserve	
	* Defining fiscal shock as a dummy variable for debt rule adoption
		gen dr_adoption = 0
		replace dr_adoption = 1 if drdebtrule == 1 & l.drdebtrule == 0	
		*replace dr_adoption = 1 if fiscal_rule_any == 1 & L.fiscal_rule_any == 0 
	* Mark the sample before running lpirf to flag the exact sample used
		mark sample_flag	
	* Estimating Local Projections (LPIRF)
		lpirf ipriv_gdp, lags(1/2) step(4) exog(dr_adoption l.spreads l.ggdy l.gdp_cc) dfk small vce(robust) level(90)
		outreg2 using "$output\LPIRF.xls", replace 
	* Save the IRF results, replacing if necessary	
		irf set lpirf.irf, replace
	* Create IRF results using the label "IRF"
		irf create IRF
	*Plot the IRF - Private Investment
		irf graph dm, irf(IRF) impulse(dr_adoption) response(ipriv_gdp) ///
		yline(0, lcolor(black) lpattern(dash)) xlabel(0(1)3) ///
		ylabel(-4(1)4) xtitle("Time Horizon (Years)") ///
		ytitle("% GDP") ///
		legend(off)	level(90)
	* Restore the original dataset
		restore	
************************************************************************************
* Figure 10. LOCAL PROJECTIONS using private investment as dependent variable (5 STEPS)
************************************************************************************	
	* Load the main database
	    use "$input/fiscal_rules_macro_variables_2000_2019_readytouse.dta", replace
	* Preserve the dataset to restore it later	
		preserve	
	* Defining fiscal shock as a dummy variable for debt rule adoption
		gen dr_adoption = 0
		replace dr_adoption = 1 if drdebtrule == 1 & l.drdebtrule == 0	
		*replace dr_adoption = 1 if fiscal_rule_any == 1 & L.fiscal_rule_any == 0 
	* Mark the sample before running lpirf to flag the exact sample used
		mark sample_flag	
	* Estimating Local Projections (LPIRF)
		lpirf ipriv_gdp, lags(1/2) step(5) exog(dr_adoption l.spreads l.ggdy l.gdp_cc) dfk small vce(robust) level(90)
		outreg2 using "$output\LPIRF.xls", replace 
	* Save the IRF results, replacing if necessary	
		irf set lpirf.irf, replace
	* Create IRF results using the label "IRF"
		irf create IRF
	*Plot the IRF - Private Investment
		irf graph dm, irf(IRF) impulse(dr_adoption) response(ipriv_gdp) ///
		yline(0, lcolor(black) lpattern(dash)) xlabel(0(1)4) ///
		ylabel(-4(1)4) xtitle("Time Horizon (Years)") ///
		ytitle("% GDP") ///
		legend(off)	level(90)
	* Restore the original dataset
		restore	
************************************************************************************
* Figure 11. LOCAL PROJECTIONS using private investment as dependent variable (7 STEPS)
************************************************************************************	
	* Load the main database
	    use "$input/fiscal_rules_macro_variables_2000_2019_readytouse.dta", replace
	* Preserve the dataset to restore it later	
		preserve	
	* Defining fiscal shock as a dummy variable for debt rule adoption
		gen dr_adoption = 0
		replace dr_adoption = 1 if drdebtrule == 1 & l.drdebtrule == 0	
		*replace dr_adoption = 1 if fiscal_rule_any == 1 & L.fiscal_rule_any == 0 
	* Mark the sample before running lpirf to flag the exact sample used
		mark sample_flag	
	* Estimating Local Projections (LPIRF)
		lpirf ipriv_gdp, lags(1/2) step(7) exog(dr_adoption l.spreads l.ggdy l.gdp_cc) dfk small vce(robust) level(90)
		outreg2 using "$output\LPIRF.xls", replace 
	* Save the IRF results, replacing if necessary	
		irf set lpirf.irf, replace
	* Create IRF results using the label "IRF"
		irf create IRF
	*Plot the IRF - Private Investment
		irf graph dm, irf(IRF) impulse(dr_adoption) response(ipriv_gdp) ///
		yline(0, lcolor(black) lpattern(dash)) xlabel(0(1)6) ///
		ylabel(-4(1)4) xtitle("Time Horizon (Years)") ///
		ytitle("% GDP") ///
		legend(off)	level(90)
	* Restore the original dataset
		restore	
************************************************************************************
* Figure 12. LOCAL PROJECTIONS using private investment as dependent variable (8 STEPS)
************************************************************************************	
	* Load the main database
	    use "$input/fiscal_rules_macro_variables_2000_2019_readytouse.dta", replace
	* Preserve the dataset to restore it later	
		preserve	
	* Defining fiscal shock as a dummy variable for debt rule adoption
		gen dr_adoption = 0
		replace dr_adoption = 1 if drdebtrule == 1 & l.drdebtrule == 0	
		*replace dr_adoption = 1 if fiscal_rule_any == 1 & L.fiscal_rule_any == 0 
	* Mark the sample before running lpirf to flag the exact sample used
		mark sample_flag	
	* Estimating Local Projections (LPIRF)
		lpirf ipriv_gdp, lags(1/2) step(8) exog(dr_adoption l.spreads l.ggdy l.gdp_cc) dfk small vce(robust) level(90)
		outreg2 using "$output\LPIRF.xls", replace 
	* Save the IRF results, replacing if necessary	
		irf set lpirf.irf, replace
	* Create IRF results using the label "IRF"
		irf create IRF
	*Plot the IRF - Private Investment
		irf graph dm, irf(IRF) impulse(dr_adoption) response(ipriv_gdp) ///
		yline(0, lcolor(black) lpattern(dash)) xlabel(0(1)7) ///
		ylabel(-4(1)4) xtitle("Time Horizon (Years)") ///
		ytitle("% GDP") ///
		legend(off)	level(90)
	* Restore the original dataset
		restore			
************************************************************************************
************************************************************************************
************************************************************************************		











