# Hazard classifier (MobileNetV2 / AIDER)

Validation split: 1286 images. Class-weighted training (AIDER is 68% `normal`).

```
                    precision    recall  f1-score   support

collapsed_building      0.797     0.979     0.879        96
              fire      0.938     0.981     0.959       107
     flooded_areas      0.892     0.958     0.924        95
  traffic_incident      0.896     0.915     0.905        94
            normal      0.988     0.949     0.968       894

          accuracy                          0.952      1286
         macro avg      0.902     0.956     0.927      1286
      weighted avg      0.956     0.952     0.953      1286

```

Confusion matrix (rows=true, cols=pred), order ['collapsed_building', 'fire', 'flooded_areas', 'traffic_incident', 'normal']:

```
[[ 94   1   0   0   1]
 [  0 105   0   0   2]
 [  0   0  91   1   3]
 [  3   0   1  86   4]
 [ 21   6  10   9 848]]
```

INT8 TFLite export: `/Users/afzalamanullah/saresq/models/mobilenetv2_aider_224_int8.tflite` (2.74 MB).
