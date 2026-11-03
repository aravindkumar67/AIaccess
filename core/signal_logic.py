from .geometry import overlap_ratio

def grade_severity(frame_shape,bbox,regions,logic):
    med=overlap_ratio(frame_shape,bbox,regions.get('medium_risk',[])); high=overlap_ratio(frame_shape,bbox,regions.get('high_risk',[]))
    level='High' if high>=float(logic.get('high_overlap',.20)) else ('Medium' if med>=float(logic.get('medium_overlap',.05)) else 'Low')
    return {'severity':level,'medium_overlap_ratio':med,'high_overlap_ratio':high}
