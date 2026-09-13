# Model card

## `smoke_test.pt`

This checkpoint exists only to prove that the local data -> train -> validate ->
inference -> export pipeline works. It was trained on 5% of the training split for
one epoch at 160 px on CPU. Its validation mAP50 is approximately 0.021, so it is
**not suitable for a portfolio demonstration or production inspection**.

`smoke_test.onnx` is the matching export. It has been loaded and executed with
OpenCV DNN 4.11; its output tensor is `[1, 10, 525]` for a 160 x 160 input.

Train on the full dataset with `python scripts/train.py --epochs 100 --imgsz 640`
to create `best.pt`, then evaluate that checkpoint on the test split before using it.
