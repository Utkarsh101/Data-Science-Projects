libname Project "\\Client\C$\Users\shreyasrewagad\Data Science\Spring 2017\DataAnalysisModeling\Project\";
proc import datafile = "\\Client\C$\Users\shreyasrewagad\Data Science\Spring 2017\DataAnalysisModeling\Project\P507_HRData.csv" out = Project.data
dmbs = csv replace;
run;
data Project.data;
   set Project.Data(rename=(sales=dept)rename=(left=EXIT));
Title "Creating Dummies for Salary and Department";
DATA Project.DataDummy ;
SET Project.Data ; 

IF dept = 'sales' THEN dept_sales = 1; 
    ELSE dept_sales = 0;

IF dept = 'accounting' THEN dept_accounting = 1; 
	ELSE dept_accounting = 0;

IF dept = 'hr' THEN dept_hr = 1; 
    ELSE dept_hr = 0;

IF dept = 'technical' THEN dept_tech = 1; 
    ELSE dept_tech = 0;

IF dept = 'support' THEN dept_support = 1; 
    ELSE dept_support = 0;

IF dept = 'management' THEN dept_mgnt = 1; 
    ELSE dept_mgnt = 0;

IF dept = 'IT' THEN dept_IT = 1; 
    ELSE dept_IT = 0;

IF dept = 'product_mng' THEN dept_prodMgnt = 1; 
    ELSE dept_prodMgnt = 0;

IF dept = 'marketing' THEN dept_mkt = 1; 
    ELSE dept_mkt = 0;

IF salary = 'medium' THEN salary_mid = 1; 
    ELSE salary_mid = 0;

IF salary = 'high' THEN salary_high = 1; 
    ELSE salary_high = 0;

IF satisfaction_level >= 0 and satisfaction_level < .33 THEN sat_low = 1; 
    ELSE sat_low = 0;
IF satisfaction_level >= 0.33 and satisfaction_level < 0.66 THEN sat_mid = 1; 
    ELSE sat_mid = 0;

IF last_evaluation >= .66 and last_evaluation <= 1 THEN eval_high = 1; 
    ELSE eval_high = 0;
IF last_evaluation >= 0.33 and last_evaluation < 0.66 THEN eval_mid = 1; 
    ELSE eval_mid = 0;

IF time_spend_company >= 0 and time_spend_company < 3 THEN time_low = 1; 
    ELSE time_low = 0;
IF time_spend_company >= 3 and time_spend_company < 6 THEN time_mid = 1; 
    ELSE time_mid = 0;

run;
*LOGISTIC REGRESSION;
Title "Logit Analysis";
Title1 "Full Model";
Proc Logistic desc;
	Model EXIT = satisfaction_level last_evaluation number_project average_montly_hours time_spend_company Work_accident promotion_last_5years dept_sales dept_accounting dept_hr dept_tech dept_support dept_mgnt dept_IT dept_prodMgnt dept_mkt salary_high salary_mid/link=logit rsq stb ctable pprob=0.45;
	**Output pred=PredFull out=Project.FullModel;
run;
Title1 "Model without the strongly colinear variables - Dept<sales>";
Proc Logistic desc;
	Model EXIT = satisfaction_level last_evaluation number_project average_montly_hours time_spend_company Work_accident promotion_last_5years dept_hr dept_IT salary_high salary_mid/link=logit rsq stb ctable pprob=0.45; 
run;

data Project.Data_Intr;
SET Project.DataDummy;
sal_dept_hr = salary_high = dept_hr*salary_high;
sal_mid_hr = dept_hr*salary_mid;
eval_hr = last_evaluation * dept_hr;
sat_low_hr = sat_low*dept_hr ;
sat_low_sal_mid = sat_low*salary_mid ;
sat_low_sal_high = sat_low*salary_high ;
sat_mid_sal_mid = sat_mid*salary_mid ;
eval_high_sal_high = eval_high*salary_high ;
time_low_eval_mid = time_low*eval_mid ;
time_mid_eval_mid = time_mid*eval_mid;
run; 

*Correlation Matrix;
**ods graphics on;
**proc corr data=Project.ResModel plots=matrix(histogram);
**var satisfaction_level last_evaluation number_project average_montly_hours time_spend_company Work_accident promotion_last_5years dept_hr dept_IT dept_RnD salary_low salary_high;
**run;
**ods graphics off;

Title "Logistic with Interaction variables";
Proc Logistic desc;
	Model EXIT = satisfaction_level last_evaluation number_project average_montly_hours time_spend_company Work_accident promotion_last_5years salary_high salary_mid dept_hr*salary_high dept_hr*salary_mid sat_low*dept_hr sat_low*salary_mid sat_low*salary_high sat_mid*salary_mid eval_high*salary_high time_low*eval_mid time_mid*eval_mid/link=logit rsq stb ctable pprob=0.45; 
	Output pred=pred out=Project.Data_Final;
	run;
Title "Checking for multicollinearity";
proc reg plots=none;
	model EXIT = satisfaction_level last_evaluation number_project average_montly_hours time_spend_company Work_accident promotion_last_5years salary_high salary_mid sal_dept_hr sal_mid_hr sat_low_hr sat_low_sal_mid sat_low_sal_high sat_mid_sal_mid eval_high_sal_high time_low_eval_mid time_mid_eval_mid/ VIF TOL COLLIN;
run;


Title1 "Logit Marginal Effects";
Proc qlim data=Project.Data_Final plots=none;
	Model EXIT = satisfaction_level last_evaluation number_project average_montly_hours time_spend_company Work_accident promotion_last_5years salary_high salary_mid dept_hr*salary_high dept_hr*salary_mid sat_low*dept_hr sat_low*salary_mid sat_low*salary_high sat_mid*salary_mid eval_high*salary_high time_low*eval_mid time_mid*eval_mid/ discrete (dist=logit);
	Output out=Project.mfx marginal;
	run;

Title1 'Logit Marginal Effects';
Proc means data=Project.Mfx mean std;
	Var Meff_P2_satisfaction_level Meff_P2_last_evaluation Meff_P2_number_project Meff_P2_average_montly_hours Meff_P2_time_spend_company Meff_P2_Work_accident Meff_P2_promotion_last_5years Meff_P2_salary_high Meff_P2_salary_mid Meff_P2_salary_high_dept_hr Meff_P2_salary_mid_dept_hr Meff_P2_dept_hr_sat_low Meff_P2_salary_mid_sat_low Meff_P2_salary_high_sat_low Meff_P2_salary_mid_sat_mid Meff_P2_salary_high_eval_high Meff_P2_time_low_eval_mid Meff_P2_eval_mid_time_mid Meff_P2_eval_mid_time_mid;
	run;
Title 'Logit Predicted Probabilities';
proc means data=Project.Data_Final;
	Var Exit pred;
	run;
