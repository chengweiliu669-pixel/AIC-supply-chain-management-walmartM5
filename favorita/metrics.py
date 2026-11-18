import numpy as np

def event_metrics(actual, forecast, previous, direction, floor):
    threshold = np.maximum(floor, 0.5 * previous)
    observed = direction * (actual - previous) >= threshold
    flagged = direction * (forecast - previous) >= threshold
    tp = int((observed & flagged).sum())
    alerts = int(flagged.sum())
    events = int(observed.sum())
    return dict(tp=tp, fp=alerts-tp, fn=events-tp, alerts=alerts, events=events,
                precision=tp/alerts if alerts else None, recall=tp/events if events else None)

