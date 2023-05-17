import json
from tqdm import tqdm
import os

resource_folder = '../resources/'
data_folder = '../data/'

# split entire map into split_num * split_num grids
split_num = 2000
# osm id vocab
geom_ids = {'[PAD]': 0, '[CLS]': 1, '[MASK]': 2}
# intrinsic feature for each geographic object
geom_info = {}
# pair of text and geolocation
poi_info = {}
# pair of text and GC
utt = {}

# bbox of map
global total_max_x
global total_max_y
global total_min_x
global total_min_y
total_max_x = 0
total_min_x = 200
total_max_y = 0
total_min_y = 200

def discrete_num(nmin, scale, n):
    nmin = int(nmin * 10000)
    n = int(n * 10000)
    if (n - nmin) % scale == 0:
        return int((n - nmin) // scale) + 3
    else:
        return int((n - nmin) // scale + 1) + 3

def update_geominfo(osm_id, name, bbox, geom_type):
    """
    calculate intrinsic feature
    geom_type: 3 (aoi), 4 (road)
    """
    global total_max_x
    global total_max_y
    global total_min_x
    global total_min_y
    if not osm_id in geom_ids:
        geom_ids[osm_id] = len(geom_ids)
    geom_id = geom_ids[osm_id]
    name = name.replace(r'\N', '')
    if 'LINESTRING' in bbox:
        bbox = bbox[len('LINESTRING('): -len(')')]
    else:
        bbox = bbox[len('POLYGON(('): -len('))')]
    max_x = 0
    min_x = 200
    max_y = 0
    min_y = 200
    for lxly in bbox.split(','):
        lx, ly = lxly.split(' ')
        lx = float(lx)
        ly = float(ly)
        if lx > max_x:
            max_x = lx
        if lx < min_x:
            min_x = lx
        if ly > max_y:
            max_y = ly
        if ly < min_y:
            min_y = ly
    total_max_x = max(max_x, total_max_x)
    total_max_y = max(max_y, total_max_y)
    total_min_x = min(min_x, total_min_x)
    total_min_y = min(min_y, total_min_y)
    bbox = [min_x, min_y, max_x, max_y]
    geom_info[osm_id] = [geom_id, name, bbox, geom_type]

# read open street map aoi in Hangzhou
for line in tqdm(open(os.path.join(resource_folder, 'hz_aoi.txt')), desc='reading aoi info'):
    line = line.strip('\n')
    osm_id, name, bbox = line.split('\t')
    update_geominfo(osm_id, name, bbox, 3)

# read open street map road in Hangzhou
for line in tqdm(open(os.path.join(resource_folder, 'hz_roads.txt')), desc='reading roads info'):
    line = line.strip('\n')
    osm_id, name, bbox = line.split('\t')
    update_geominfo(osm_id, name, bbox, 4)

# use bbox of entire map (all aois and roads) to calculate scale factor
scale_x = ((total_max_x - total_min_x) * 10000 + 1) / split_num
scale_y = ((total_max_y - total_min_y) * 10000 + 1) / split_num
print(total_max_x, total_min_x, scale_x)
print(total_max_y, total_min_y, scale_y)

# read text and geolocation pair
pair_idx = 0
for line in tqdm(open(os.path.join(resource_folder, 'text_location_pair.demo')), desc='reading pairs info'):
    line = line.strip('\n')
    items = line.split('')
    idx = str(pair_idx)
    text = items[0]

    # geolocation should use wgs84
    lx, ly = items[1].split(',')

    ly = float(ly)
    lx = float(lx)
    poi_info[idx] = [text, [lx, ly]]
    utt[idx] = [[], [], [], [], [], text, idx, items[1]]
    pair_idx += 1


sign = lambda x: 1 if x > 0 else -1
def update_utt(idx, osmid, rel_type):
    """
    calculate geographic context, which is used in geographic encoder 
    """
    global total_max_x
    global total_max_y
    global total_min_x
    global total_min_y

    geom_id, name, bbox, geom_type = geom_info[osmid]
    text = poi_info[idx][0]
    x, y = poi_info[idx][1]
    lx1, ly1, lx2, ly2 = bbox
    utt[idx][0].append(geom_id)
    utt[idx][1].append(geom_type)
    utt[idx][2].append(rel_type)
    utt[idx][3].append([discrete_num(total_min_x, scale_x, lx1),\
                        discrete_num(total_min_y, scale_y, ly1),\
                        discrete_num(total_min_x, scale_x, lx2),\
                        discrete_num(total_min_y, scale_y, ly2)])
    if lx1 == lx2:
        lx2 += 0.0001
    if ly1 == ly2:
        ly2 += 0.0001
    utt[idx][4].append([sign(x - lx1) * min(20, int(abs(x - lx1) / abs(lx1 - lx2) * 10)) + 23,\
                        sign(y - ly1) *  min(20, int(abs(y - ly1) / abs(ly1 - ly2) * 10)) + 23,\
                        sign(x - lx2) *  min(20, int(abs(x - lx2) / abs(lx1 - lx2) * 10)) + 23,\
                        sign(y - ly2) * min(20, int(abs(y - ly1) / abs(ly1 - ly2) * 10)) + 23])

# use relations to calculate geographic context
inaoi = set()
for line in tqdm(open(os.path.join(resource_folder, 'location_in_aoi.demo')), desc='generate data'):
    line = line.strip('\n')
    idx, osmid = line.split('\t')
    inaoi.add(idx + osmid)
    update_utt(idx, osmid, 3)

for line in tqdm(open(os.path.join(resource_folder, 'location_near_aoi.demo')), desc='generate data'):
    line = line.strip('\n')
    idx, osmid = line.split('\t')
    if idx + osmid in inaoi: continue
    update_utt(idx, osmid, 4)

for line in tqdm(open(os.path.join(resource_folder, 'location_near_road.demo')), desc='generate data'):
    line = line.strip('\n')
    idx, osmid = line.split('\t')
    if idx + osmid in inaoi: continue
    update_utt(idx, osmid, 4)

print('geom_ids size', len(geom_ids))

hz_ids = []
out = open(os.path.join(data_folder, 'text_location.jsonl'), 'w')
# for generate test samples
for i in range(1000):
    for idx in utt:
        hz_ids.append(idx)
        out.write(json.dumps(utt[idx], ensure_ascii=False) + '\n')
out.close()

