"""Calibration (slope + calibration-in-the-large intercept) for the Aim-3 handling-strategy
mortality model, per strategy. Calibration is reported because the paper discusses risk
shifts. Out-of-fold predictions, same model as 12_aim3."""
import warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore"); np.seterr(all="ignore")
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from config import INT, TAB, ICU, IT, EYE_SCORE, MOTOR_SCORE
from lib_gcs_strategies import total_gcs

cohort = pd.read_parquet(INT/"cohort.parquet"); cohort = cohort[cohort.first_stay]
neuro = cohort[cohort.phenotype != "comparator"]
icu = pd.read_csv(ICU/"icustays.csv.gz", usecols=["stay_id","intime"], parse_dates=["intime"])
chart = pd.read_parquet(INT/"chart_neuro.parquet").merge(icu, on="stay_id")
chart["hr"] = (chart.charttime-chart.intime).dt.total_seconds()/3600
d1 = chart[(chart.hr>=0)&(chart.hr<=24)]
eye=d1[d1.itemid==IT["gcs_eye"]].assign(eye=lambda x:x["value"].map(EYE_SCORE))
mot=d1[d1.itemid==IT["gcs_motor"]].assign(motor=lambda x:x["value"].map(MOTOR_SCORE))
vrb=d1[d1.itemid==IT["gcs_verbal"]].rename(columns={"value":"verbal_raw"})
trip=(eye[["stay_id","charttime","eye"]].merge(mot[["stay_id","charttime","motor"]],on=["stay_id","charttime"])
      .merge(vrb[["stay_id","charttime","verbal_raw"]],on=["stay_id","charttime"]).dropna())
trip["em"]=trip.eye+trip.motor
worst=trip.sort_values(["stay_id","em","charttime"]).groupby("stay_id").first().reset_index()
X=neuro.merge(worst,on="stay_id").set_index("stay_id")
inf=pd.read_parquet(INT/"infusions.parquet"); vent=pd.read_parquet(INT/"vent.parquet")
X["vaso"]=X.index.isin(inf[inf.cls=="vasopressor"].stay_id).astype(int)
X["vent"]=X.index.isin(vent.stay_id).astype(int)
y=X["hospital_expire_flag"].astype(int).values


def oof_pred(tg):
    m=~np.isnan(tg); feats=np.column_stack([tg,X.age.values,X.vaso.values,X.vent.values])[m]; yy=y[m]
    oof=np.zeros(len(yy))
    for tr,te in StratifiedKFold(5,shuffle=True,random_state=0).split(feats,yy):
        clf=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000)).fit(feats[tr],yy[tr])
        oof[te]=clf.predict_proba(feats[te])[:,1]
    return yy,oof


def calib(yy,p):
    p=np.clip(p,1e-6,1-1e-6); lp=np.log(p/(1-p))
    slope=sm.Logit(yy,sm.add_constant(lp)).fit(disp=0).params[1]          # calibration slope
    intercept=sm.Logit(yy,np.ones((len(yy),1)),offset=lp).fit(disp=0).params[0]  # calibration-in-the-large
    return round(slope,2), round(intercept,2)


rows=[]
for name,strat in [("Complete-case","drop"),("Impute=1","impute1"),("E+M (T)","gcs_t"),
                   ("Brennan","kramer")]:
    tg=np.array([total_gcs(dict(eye=e,motor=m,verbal_raw=v),strat) for e,m,v in zip(X.eye,X.motor,X.verbal_raw)],float)
    yy,p=oof_pred(tg); s,i=calib(yy,p)
    rows.append(dict(strategy=name,n=len(yy),calibration_slope=s,calibration_intercept=i))
res=pd.DataFrame(rows); res.to_csv(TAB/"table_calibration.csv",index=False)
print(res.to_string(index=False))
