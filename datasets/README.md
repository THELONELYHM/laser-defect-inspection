# Dataset notes

This project uses **NEU-DET**, the detection subset of the Northeastern University
surface defect database. It contains 1,800 grayscale steel-surface images at
200 x 200 pixels and Pascal VOC bounding-box annotations for six classes.

- Official page: https://faculty.neu.edu.cn/songkc/en/zdylm/263270/list/index.htm
- Official Google Drive file: https://drive.google.com/file/d/1qrdZlaDi272eA79b0uCwwqPrm2Q_WI3k/view
- Baidu Pan is also linked from the official page (code: `pmqx`).

The dataset belongs to its original authors and is not covered by this project's
MIT license. Follow the source's research-use and citation requirements.

Run `python scripts/download_neu_det.py --prepare` to download, extract and
convert it. The generated split is stratified per image class with seed 42:

- train: 1,260 images
- validation: 270 images
- test: 270 images

The converter removes three exact duplicate bounding boxes from the source XML.
The prepared split contains 4,186 labeled objects and passes `audit_dataset.py`
without missing pairs or invalid normalized coordinates.
