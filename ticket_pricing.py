"""Philippine concert ticket pricing: what drives the price, and how well can it be predicted?

NOTE: the dataset is SYNTHETIC. Every row is labelled data_origin = SYNTHETIC_SIMULATED. The results describe how
well models recover a simulated pricing process, not real ticket-market behaviour.

Pipeline
  1. Load and validate the data (one event = 8 ticket tiers).
  2. Exploratory charts.
  3. Features known before tickets go on sale (the simulated demand index is excluded; it is tested separately).
  4. Models: Random Forest, Extra Trees, Gradient Boosting, Neural Network, plus an ensemble of the best three.
     Benchmark: median price per ticket tier. Every model is trained on log(price).
  5. Hyperparameters tuned with 5-fold cross-validation grouped by event on the training events only.
  6. Model chosen by cross-validated MAE. The held-out events are scored once.
  7. Extras: feature ablation, out-of-time check, permutation importance, error by tier, 80% prediction intervals.

Run from the project root:  python src/ticket_pricing.py
"""
from pathlib import Path
import json
import warnings

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import (ExtraTreesRegressor, HistGradientBoostingRegressor,
                              RandomForestRegressor, VotingRegressor)
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, RandomizedSearchCV, cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'philippine_concert_tickets.csv'
OUT, FIG = ROOT / 'outputs', ROOT / 'outputs' / 'figures'
TARGET, GROUP = 'ticket_price_php', 'event_id'
SEED, CV_FOLDS, N_ITER = 42, 5, 10
TIER_ORDER = ['General Admission', 'Upper Box', 'Regular Seated', 'Lower Box',
              'Premium Seated', 'VIP', 'VIP Soundcheck', 'VVIP']
CATEGORICAL = ['artist_category', 'venue', 'city_or_province', 'ticketing_fees']
NUMERIC = ['tier_rank', 'log_capacity', 'artist_popularity_index', 'days_before_event_listing',
           'weekend_event', 'year', 'month_sin', 'month_cos', 'day_of_week']


# ----------------------------------------------------------------------------- data
def load():
    df = pd.read_csv(DATA, parse_dates=['concert_date'])
    assert df.record_id.is_unique and not df.isna().any().any()
    assert df.groupby(GROUP).size().eq(len(TIER_ORDER)).all(), 'expected 8 tiers per event'
    assert set(df.ticket_tier) == set(TIER_ORDER)
    return df


def featurize(df, with_demand=False):
    X = pd.DataFrame(index=df.index)
    X['tier_rank'] = df.ticket_tier.map({t: i for i, t in enumerate(TIER_ORDER)})
    X['log_capacity'] = np.log(df.venue_capacity_est)
    X['artist_popularity_index'] = df.artist_popularity_index
    X['days_before_event_listing'] = df.days_before_event_listing
    X['weekend_event'] = df.weekend_event
    X['year'] = df.concert_date.dt.year
    X['month_sin'] = np.sin(2 * np.pi * df.concert_date.dt.month / 12)
    X['month_cos'] = np.cos(2 * np.pi * df.concert_date.dt.month / 12)
    X['day_of_week'] = df.concert_date.dt.dayofweek
    for c in CATEGORICAL:
        X[c] = df[c].astype(str)
    if with_demand:
        X['event_demand_index_simulated'] = df.event_demand_index_simulated
    return X


class TierMedian(BaseEstimator, RegressorMixin):
    """Benchmark: predict the median training price of the ticket tier."""

    def fit(self, X, y):
        self.med_ = pd.Series(np.asarray(y).ravel()).groupby(np.asarray(X['tier_rank'])).median()
        return self

    def predict(self, X):
        return X['tier_rank'].map(self.med_).to_numpy()


# ----------------------------------------------------------------------------- models
def wrap(estimator, numeric, scale=False):
    pre = ColumnTransformer([
        ('num', StandardScaler() if scale else 'passthrough', numeric),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL)])
    target = Pipeline([('log', FunctionTransformer(np.log, np.exp, check_inverse=False)),
                       ('std', StandardScaler())])
    return TransformedTargetRegressor(regressor=Pipeline([('pre', pre), ('model', estimator)]),
                                      transformer=target)


def candidates(numeric):
    p = 'regressor__model__'
    return {
        'Random Forest': (wrap(RandomForestRegressor(random_state=SEED, n_jobs=1), numeric),
                          {p + 'n_estimators': randint(200, 500), p + 'max_depth': randint(8, 30),
                           p + 'min_samples_leaf': randint(1, 8), p + 'max_features': [.5, .7, 1.0]}),
        'Extra Trees': (wrap(ExtraTreesRegressor(random_state=SEED, n_jobs=1), numeric),
                        {p + 'n_estimators': randint(200, 500), p + 'max_depth': randint(8, 30),
                         p + 'min_samples_leaf': randint(1, 8), p + 'max_features': [.5, .7, 1.0]}),
        'Gradient Boosting': (wrap(HistGradientBoostingRegressor(random_state=SEED), numeric),
                              {p + 'learning_rate': loguniform(.02, .2), p + 'max_iter': randint(200, 700),
                               p + 'max_depth': randint(3, 10), p + 'min_samples_leaf': randint(10, 60),
                               p + 'l2_regularization': loguniform(1e-3, 10),
                               p + 'max_leaf_nodes': randint(15, 64)}),
        'Neural Network': (wrap(MLPRegressor(max_iter=300, early_stopping=True, random_state=SEED),
                                numeric, scale=True),
                           {p + 'hidden_layer_sizes': [(64,), (128, 64), (128, 64, 32)],
                            p + 'alpha': loguniform(1e-5, 1e-2),
                            p + 'learning_rate_init': loguniform(5e-4, 5e-3)}),
    }


def metrics(y_true, pred):
    return {'MAE': mean_absolute_error(y_true, pred), 'RMSE': np.sqrt(mean_squared_error(y_true, pred)),
            'R2': r2_score(y_true, pred), 'MAPE': float(np.mean(np.abs(y_true - pred) / y_true) * 100)}


def tune_all(X_train, y_train, g_train, numeric):
    cv = GroupKFold(n_splits=CV_FOLDS)
    tuned, rows = {}, []
    for name, (est, space) in candidates(numeric).items():
        print(f'Tuning {name} ({N_ITER} configs x {CV_FOLDS} grouped folds)...', flush=True)
        s = RandomizedSearchCV(est, space, n_iter=N_ITER, cv=cv, scoring='neg_mean_absolute_error',
                               n_jobs=2, random_state=SEED, refit=True)
        s.fit(X_train, y_train, groups=g_train)
        tuned[name] = s.best_estimator_
        rows.append({'Model': name, 'CV MAE': -s.best_score_,
                     'CV MAE Std': s.cv_results_['std_test_score'][s.best_index_]})
        print(f'  CV MAE PHP {-s.best_score_:,.0f}', flush=True)
    top3 = [r['Model'] for r in sorted(rows, key=lambda r: r['CV MAE'])][:3]
    ens = VotingRegressor([(n, tuned[n]) for n in top3])
    sc = -cross_val_score(ens, X_train, y_train, groups=g_train, cv=cv, scoring='neg_mean_absolute_error', n_jobs=2)
    ens.fit(X_train, y_train)
    tuned['Ensemble (top 3)'] = ens
    rows.append({'Model': f'Ensemble (top 3)', 'CV MAE': sc.mean(), 'CV MAE Std': sc.std()})
    print(f'Ensemble of {top3}: CV MAE PHP {sc.mean():,.0f}', flush=True)
    bm = -cross_val_score(TierMedian(), X_train, y_train, groups=g_train, cv=cv, scoring='neg_mean_absolute_error')
    rows.append({'Model': 'Benchmark: tier median', 'CV MAE': bm.mean(), 'CV MAE Std': bm.std()})
    return tuned, pd.DataFrame(rows).sort_values('CV MAE').reset_index(drop=True), top3


# ----------------------------------------------------------------------------- main
def main():
    OUT.mkdir(exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    df = load()
    origin = df.data_origin.unique().tolist()
    print(f'{len(df):,} rows, {df[GROUP].nunique():,} events x {len(TIER_ORDER)} tiers; data_origin = {origin}')
    y = df[TARGET].to_numpy()
    X = featurize(df)
    groups = df[GROUP]

    plt.rcParams.update({'figure.dpi': 120, 'axes.spines.top': False, 'axes.spines.right': False})
    blue, red, grey = '#1f5fa8', '#c0392b', '#8a8a8a'

    # ---- exploratory charts
    fig, ax = plt.subplots(figsize=(9, 4.8))
    data = [df.loc[df.ticket_tier == t, TARGET] for t in TIER_ORDER]
    ax.boxplot(data, vert=False, tick_labels=TIER_ORDER, showfliers=False, patch_artist=True,
               boxprops=dict(facecolor='#cfe0f3', color=blue), medianprops=dict(color=red))
    ax.set_xlabel('Ticket price (PHP)'); ax.set_title('Price by ticket tier (synthetic data)', loc='left')
    fig.tight_layout(); fig.savefig(FIG / '01_price_by_tier.png'); plt.close(fig)

    ga = df[df.ticket_tier == 'General Admission']
    fig, ax = plt.subplots(figsize=(9, 5))
    for cat, g in ga.groupby('artist_category'):
        ax.scatter(g.artist_popularity_index, g[TARGET], s=10, alpha=.5, label=cat)
    ax.set_xlabel('Artist popularity index'); ax.set_ylabel('General Admission price (PHP)')
    ax.set_title('Popularity drives price; category shifts the level', loc='left')
    ax.legend(frameon=False, fontsize=8, ncol=2, markerscale=2)
    fig.tight_layout(); fig.savefig(FIG / '02_popularity_vs_price.png'); plt.close(fig)

    # ---- split: whole events held out, never used for tuning or selection
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=SEED).split(X, y, groups=groups))
    X_tr, X_te, y_tr, y_te, g_tr = X.iloc[tr], X.iloc[te], y[tr], y[te], groups.iloc[tr]
    print(f'Train/test records: {len(tr):,}/{len(te):,}; events: {g_tr.nunique():,}/{groups.iloc[te].nunique():,}')

    # ---- tune + cross-validate
    tuned, cv_table, top3 = tune_all(X_tr, y_tr, g_tr, NUMERIC)
    print('\nGROUPED CROSS-VALIDATION (training events only)')
    print(cv_table.round(1).to_string(index=False))
    best = cv_table.loc[~cv_table.Model.str.startswith('Benchmark'), 'Model'].iloc[0]
    print(f'\nSelected by CV MAE: {best}')

    # ---- one-time test evaluation
    preds = {n: m.predict(X_te) for n, m in tuned.items()}
    preds['Benchmark: tier median'] = TierMedian().fit(X_tr, y_tr).predict(X_te)
    test = pd.DataFrame({n: metrics(y_te, p) for n, p in preds.items()}).T.sort_values('MAE')
    print('\nTEST SET (held-out events, scored once, not used for selection)')
    print(test.round(3).to_string())

    # ---- feature ablation, using the best single model's tuned configuration (the ensemble cannot be re-wired)
    single = cv_table.loc[~cv_table.Model.str.startswith(('Benchmark', 'Ensemble')), 'Model'].iloc[0]
    best_model = tuned[best]

    def ablate(num, cats, frame):
        m = clone(tuned[single])
        m.regressor.set_params(pre__transformers=[
            ('num', 'passthrough', num),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cats)])
        m.fit(frame.iloc[tr], y_tr)
        return metrics(y_te, m.predict(frame.iloc[te]))

    demand = featurize(df, with_demand=True)
    ablation = {
        'All pre-sale features (main)': ablate(NUMERIC, CATEGORICAL, X),
        'Plus simulated demand index': ablate(NUMERIC + ['event_demand_index_simulated'], CATEGORICAL, demand),
        'Drop artist popularity': ablate([c for c in NUMERIC if c != 'artist_popularity_index'], CATEGORICAL, X),
        'Drop artist category': ablate(NUMERIC, [c for c in CATEGORICAL if c != 'artist_category'], X),
        'Drop venue and capacity': ablate([c for c in NUMERIC if c != 'log_capacity'],
                                          [c for c in CATEGORICAL if c not in ('venue', 'city_or_province')], X),
        'Ticket tier only': ablate(['tier_rank'], [], X),
    }
    ablation = pd.DataFrame(ablation).T
    print(f'\nFEATURE ABLATION ({single} configuration, held-out events)')
    print(ablation.round(3).to_string())

    before = (df.concert_date < '2026-01-01').to_numpy()
    oot = {}
    m = clone(best_model).fit(X[before], y[before])
    oot[best] = metrics(y[~before], m.predict(X[~before]))
    oot['Benchmark: tier median'] = metrics(y[~before], TierMedian().fit(X[before], y[before]).predict(X[~before]))
    oot = pd.DataFrame(oot).T
    print(f'\nOUT-OF-TIME CHECK: train on events before 2026 ({before.sum():,} rows), test on 2026 ({(~before).sum():,} rows)')
    print(oot.round(3).to_string())

    # ---- error breakdown for the selected model
    pred = preds[best]
    ev = pd.DataFrame({'tier': df.ticket_tier.iloc[te].to_numpy(), 'category': df.artist_category.iloc[te].to_numpy(),
                       'actual': y_te, 'pred': pred})
    ev['abs_err'] = (ev.actual - ev.pred).abs()
    ev['ape'] = ev.abs_err / ev.actual * 100
    by_tier = ev.groupby('tier').agg(MAE=('abs_err', 'mean'), MAPE=('ape', 'mean')).reindex(TIER_ORDER)
    by_cat = ev.groupby('category').agg(MAE=('abs_err', 'mean'), MAPE=('ape', 'mean')).sort_values('MAE')
    print('\nERROR BY TIER'); print(by_tier.round(1).to_string())

    # ---- 80% prediction intervals: raw quantile models, then conformalized quantile regression (CQR)
    def quantile_model(q):
        return wrap(HistGradientBoostingRegressor(loss='quantile', quantile=q, learning_rate=.05, max_iter=300,
                                                  max_depth=6, random_state=SEED), NUMERIC)

    def interval(models, frame):
        lo, hi = (np.log(models[q].predict(frame)) for q in (.1, .9))
        return np.minimum(lo, hi), np.maximum(lo, hi)

    full = {q: quantile_model(q).fit(X_tr, y_tr) for q in (.1, .9)}
    lo, hi = interval(full, X_te)
    raw_cov = float(np.mean((np.log(y_te) >= lo) & (np.log(y_te) <= hi)))

    fit_i, cal_i = next(GroupShuffleSplit(n_splits=1, test_size=.25, random_state=SEED).split(X_tr, groups=g_tr))
    part = {q: quantile_model(q).fit(X_tr.iloc[fit_i], y_tr[fit_i]) for q in (.1, .9)}
    clo, chi = interval(part, X_tr.iloc[cal_i])
    ycal = np.log(y_tr[cal_i])
    score = np.maximum(clo - ycal, ycal - chi)
    adj = np.quantile(score, min(1, np.ceil((len(score) + 1) * .8) / len(score)))
    lo, hi = interval(part, X_te)
    lo, hi = lo - adj, hi + adj
    coverage = float(np.mean((np.log(y_te) >= lo) & (np.log(y_te) <= hi)))
    width = float(np.mean((np.exp(hi) - np.exp(lo)) / pred * 100))
    print(f'\n80% INTERVALS on held-out events: raw quantile models cover {raw_cov:.0%}; '
          f'conformalized cover {coverage:.0%} (target 80%), mean width {width:.0f}% of the prediction')

    # ---- permutation importance (grouped by original feature, on held-out events)
    perm = permutation_importance(best_model, X_te, y_te, scoring='neg_mean_absolute_error',
                                  n_repeats=5, random_state=SEED, n_jobs=2)
    imp = pd.DataFrame({'Feature': X_te.columns, 'MAE increase (PHP)': perm.importances_mean,
                        'Std': perm.importances_std}).sort_values('MAE increase (PHP)', ascending=False)
    print('\nPERMUTATION IMPORTANCE'); print(imp.round(1).to_string(index=False))

    # ---- save tables
    cv_table.round(2).to_csv(OUT / 'cv_summary.csv', index=False)
    test.round(4).to_csv(OUT / 'test_summary.csv')
    ablation.round(4).to_csv(OUT / 'ablation.csv')
    oot.round(4).to_csv(OUT / 'out_of_time.csv')
    by_tier.round(2).to_csv(OUT / 'error_by_tier.csv')
    by_cat.round(2).to_csv(OUT / 'error_by_category.csv')
    imp.round(3).to_csv(OUT / 'permutation_importance.csv', index=False)
    json.dump({'data_origin': origin, 'rows': len(df), 'events': int(df[GROUP].nunique()), 'selected_model': best,
               'ensemble_members': top3, 'interval_coverage_80_raw': raw_cov, 'interval_coverage_80_conformal': coverage,
               'interval_width_pct_conformal': width},
              open(OUT / 'run_info.json', 'w'), indent=2)
    joblib.dump(best_model, OUT / 'best_model.joblib')

    # ---- figures
    order = cv_table.sort_values('CV MAE', ascending=False)
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    colors = [grey if m.startswith('Benchmark') else (red if m == best else blue) for m in order.Model]
    ax.barh(order.Model, order['CV MAE'], xerr=order['CV MAE Std'], color=colors, error_kw=dict(lw=1, capsize=2))
    ax.set_xlabel('Cross-validated MAE (PHP)'); ax.set_title('Model comparison, grouped 5-fold CV', loc='left')
    fig.tight_layout(); fig.savefig(FIG / '03_model_comparison.png'); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(y_te, pred, s=6, alpha=.3, color=blue)
    lim = [0, max(y_te.max(), pred.max()) * 1.02]
    ax.plot(lim, lim, color=red, ls='--', lw=1); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel('Actual price (PHP)'); ax.set_ylabel('Predicted price (PHP)')
    ax.set_title(f'{best}: held-out events', loc='left')
    fig.tight_layout(); fig.savefig(FIG / '04_actual_vs_predicted.png'); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(by_tier.index[::-1], by_tier.MAPE[::-1], color=blue)
    ax.set_xlabel('Mean absolute percentage error (%)'); ax.set_title('Error by ticket tier', loc='left')
    fig.tight_layout(); fig.savefig(FIG / '05_error_by_tier.png'); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    top = imp.head(9)[::-1]
    ax.barh(top.Feature, top['MAE increase (PHP)'], xerr=top.Std, color=blue, error_kw=dict(lw=1, capsize=2))
    ax.set_xlabel('Increase in MAE when the feature is shuffled (PHP)')
    ax.set_title('Permutation importance (held-out events)', loc='left')
    fig.tight_layout(); fig.savefig(FIG / '06_permutation_importance.png'); plt.close(fig)

    sample = X_te.sample(min(1500, len(X_te)), random_state=SEED).astype({c: float for c in NUMERIC})
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    PartialDependenceDisplay.from_estimator(best_model, sample, ['artist_popularity_index', 'log_capacity'],
                                            grid_resolution=15, ax=ax)
    for a in ax:
        a.set_ylabel('Predicted price (PHP)')
    fig.suptitle('Partial dependence', x=.06, ha='left')
    fig.tight_layout(); fig.savefig(FIG / '07_partial_dependence.png'); plt.close(fig)

    rng = np.random.RandomState(SEED)
    pick = np.sort(rng.choice(len(y_te), 70, replace=False))
    pick = pick[np.argsort(y_te[pick])]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    xs = np.arange(len(pick))
    ax.vlines(xs, np.exp(lo[pick]), np.exp(hi[pick]), color=red, alpha=.35, lw=3, label='80% interval')
    ax.scatter(xs, y_te[pick], s=14, color=blue, zorder=3, label='Actual price')
    ax.set_xlabel('70 random held-out tickets, sorted by actual price'); ax.set_ylabel('PHP')
    ax.set_title(f'Prediction intervals (conformalized, {coverage:.0%} coverage on held-out events)', loc='left')
    ax.legend(frameon=False); fig.tight_layout(); fig.savefig(FIG / '08_prediction_intervals.png'); plt.close(fig)


if __name__ == '__main__':
    main()
