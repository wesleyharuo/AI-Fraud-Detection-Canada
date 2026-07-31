from evidently.report import Report
from evidently.metrics import (
    DataDriftPreset,
    DataQualityPreset,
    ClassificationQualityMetric,
)


def build_drift_report(ref_df, cur_df, target_col):
    report = Report(
        metrics=[
            DataQualityPreset(),
            DataDriftPreset(),
            ClassificationQualityMetric(
                target=target_col, prediction_probas="proba", prediction="pred"
            ),
        ]
    )
    report.run(reference_data=ref_df, current_data=cur_df)
    return report
