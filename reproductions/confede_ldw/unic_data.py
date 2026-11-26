from functools import lru_cache
from pathlib import Path
import csv
import io
import re
import xml.etree.ElementTree as ET
import zipfile
import numpy as np
from triagedmsa.mmap_pickle import load_mmap_pickle
from reproductions.confede.data import SIMSData

FEATURES = Path('data/UniC.pkl')
ANNOTATIONS = Path('data/UniC_labels.csv')
COUNTS = dict(train=582, valid=191, test=191)
NS = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}


def annotation_rows(path=ANNOTATIONS):
    path = Path(path)
    if not zipfile.is_zipfile(path):
        text = path.read_text(encoding='utf-8-sig')
        dialect = csv.Sniffer().sniff(text[:8192])
        return list(csv.DictReader(io.StringIO(text), dialect=dialect))
    with zipfile.ZipFile(path) as archive:
        strings = []
        if 'xl/sharedStrings.xml' in archive.namelist():
            strings = [''.join(t.text or '' for t in row.findall('.//s:t', NS))
                       for row in ET.fromstring(archive.read('xl/sharedStrings.xml')).findall('s:si', NS)]
        workbook = ET.fromstring(archive.read('xl/workbook.xml'))
        sheets = workbook.findall('s:sheets/s:sheet', NS)
        if len(sheets) != 1:
            raise ValueError('Expected one UniC sentiment annotation sheet')
        relation = sheets[0].get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id')
        rels = ET.fromstring(archive.read('xl/_rels/workbook.xml.rels'))
        target = next(r.get('Target') for r in rels if r.get('Id') == relation)
        target = target.lstrip('/') if target.startswith('/') else 'xl/' + target
        rows = []
        for row in ET.fromstring(archive.read(target)).findall('s:sheetData/s:row', NS):
            values = {}
            for cell in row.findall('s:c', NS):
                value = cell.find('s:v', NS)
                text = value.text if value is not None else ''
                if cell.get('t') == 's':
                    text = strings[int(text)]
                elif cell.get('t') == 'inlineStr':
                    text = ''.join(t.text or '' for t in cell.findall('.//s:t', NS))
                values[re.sub(r'\d', '', cell.get('r'))] = text
            if any(values.values()):
                rows.append(values)
        header = rows[0]
        return [{name: row.get(column, '') for column, name in header.items()} for row in rows[1:]]


def annotation_index(path=ANNOTATIONS):
    result = {}
    for row in annotation_rows(path):
        clip = str(row['clip_id']).strip().zfill(6)
        key = str(row['video_id']).strip() + '$_$' + clip
        if key in result:
            raise ValueError('Duplicate annotation ID: ' + key)
        labels = {mode: float(row['label' + ('' if mode == 'M' else '_' + mode)]) for mode in 'MTAV'}
        if not all(np.isfinite(v) and v in (-1., -.5, 0., .5, 1.) for v in labels.values()):
            raise ValueError('Invalid sentiment labels: ' + key)
        result[key] = dict(row=row, labels=labels)
    return result


@lru_cache(maxsize=2)
def feature_data(path):
    return load_mmap_pickle(path)


class UniCData(SIMSData):
    def __init__(self, path=FEATURES, split='train', limit=None, sampler_mode='upstream', annotations=None, **kwargs):
        if split not in COUNTS:
            raise ValueError(split)
        self.split = split
        self.data = feature_data(str(Path(path).resolve()))[split]
        self.size = len(self.data['id']) if limit is None else min(limit, len(self.data['id']))
        annotations = annotation_index(annotations) if annotations is not None else annotation_index()
        ids = [str(v) for v in self.data['id'][:self.size]]
        for key in ids:
            if annotations[key]['row']['mode'].strip() != split:
                raise ValueError('Annotation/feature split mismatch: ' + key)
        self.labels = {m: np.asarray([annotations[key]['labels'][m] for key in ids], dtype=np.float32) for m in 'MTAV'}
        self.sampler_mode = sampler_mode
        self.dataset = 'unic'
        self.pools = None

    @property
    def raw_M(self):
        return self.labels['M']


def normalize_text(value):
    return ' '.join(str(value).split())

