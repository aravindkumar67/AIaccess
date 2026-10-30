from fallwatch.geometry import point_in_polygon,bbox_center

def test_point_inside_normalized_polygon():
    poly=[[.1,.1],[.9,.1],[.9,.9],[.1,.9]]
    assert point_in_polygon((50,50),poly,(100,100,3))
    assert not point_in_polygon((5,5),poly,(100,100,3))

def test_bbox_center(): assert bbox_center((10,20,30,40))==(20,30)
