*   I used `mi-fgsm` to pass strong baseline.
*   I used `ensemble models + mi-fgsm` to pass boss baseline.
*   I used `ensemble models + mdii-fgsm` to pass boss baseline as well.

|      model      | public | private |
| :-------------: | :----: | :-----: |
|     mi-fgsm     | 0.210  |  0.200  |
|  ensemble + mi  | 0.120  |  0.170  |
| ensemble + mdii | 0.060  |  0.080  |