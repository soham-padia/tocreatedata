# Mechanistic Results Summary

## pro_human

- Root: `outputs/mechanistic_dataset/pro_human_beam15_20260503_233034`
- Direction: `global`
- Direction sign: `1.0`
- Search mode: `beam`
- Rows: `1000`
- Segmentation failures: `0`

### Score Stats
- mean=10.8214 median=10.7673 min=9.9164 max=12.5972 stdev=0.5631
- p10=10.1184 p25=10.3636 p75=11.2320 p90=11.5697

### Delta-Norm Stats
- mean=131.2189 median=131.6491 min=93.2071 max=148.3324 stdev=6.9424

### Top-Slice Means
- top_1: 12.5972
- top_10: 12.3358
- top_50: 12.0861
- top_100: 11.8968
- top_250: 11.5841
- top_500: 11.2890
- top_1000: 10.8214

### Length Profile
- length 6: count=1 mean_score=10.1512 mean_delta_norm=95.4399
- length 7: count=1 mean_score=10.2434 mean_delta_norm=98.2970
- length 8: count=1 mean_score=10.3602 mean_delta_norm=125.2550
- length 9: count=17 mean_score=10.3179 mean_delta_norm=121.4880
- length 10: count=39 mean_score=10.2868 mean_delta_norm=121.2956
- length 11: count=62 mean_score=10.2864 mean_delta_norm=123.3149
- length 12: count=140 mean_score=10.6061 mean_delta_norm=127.1416
- length 13: count=241 mean_score=10.9573 mean_delta_norm=131.2299
- length 14: count=253 mean_score=11.0256 mean_delta_norm=134.1942
- length 15: count=245 mean_score=10.8624 mean_delta_norm=135.0252

### Repetition
- mean unique units per row: 3.983
- mean repeat fraction: 0.696
- mean max-unit fraction: 0.427

### Top Units
- cooperation: 5236
- human dignity: 4449
- truthfulness: 2391
- solidarity: 1196
- human welfare: 4

### Top Bigrams
- cooperation -> human dignity: 2594
- cooperation -> cooperation: 2053
- human dignity -> cooperation: 1943
- human dignity -> human dignity: 1130
- truthfulness -> solidarity: 1039
- human dignity -> truthfulness: 981
- solidarity -> cooperation: 973
- truthfulness -> human dignity: 723
- cooperation -> truthfulness: 256
- truthfulness -> cooperation: 254

### Top Trigrams
- human dignity -> cooperation -> human dignity: 1315
- cooperation -> human dignity -> cooperation: 1306
- cooperation -> cooperation -> human dignity: 1180
- truthfulness -> solidarity -> cooperation: 972
- solidarity -> cooperation -> cooperation: 966
- cooperation -> cooperation -> cooperation: 763
- cooperation -> human dignity -> truthfulness: 594
- cooperation -> human dignity -> human dignity: 579
- human dignity -> truthfulness -> human dignity: 559
- human dignity -> human dignity -> cooperation: 399

### Top Examples
- rank 1: score=12.5972 len=14 units=['truthfulness', 'solidarity', 'cooperation', 'cooperation', 'cooperation', 'human dignity', 'cooperation', 'human dignity', 'truthfulness', 'human dignity', 'cooperation', 'human dignity', 'cooperation', 'human dignity'] text=`truthfulness solidarity cooperation cooperation cooperation human dignity cooperation human dignity truthfulness human dignity cooperation human dignity cooperation human dignity`
- rank 2: score=12.5921 len=13 units=['truthfulness', 'solidarity', 'cooperation', 'cooperation', 'cooperation', 'human dignity', 'cooperation', 'human dignity', 'truthfulness', 'human dignity', 'cooperation', 'human dignity', 'human dignity'] text=`truthfulness solidarity cooperation cooperation cooperation human dignity cooperation human dignity truthfulness human dignity cooperation human dignity human dignity`
- rank 3: score=12.2990 len=15 units=['truthfulness', 'solidarity', 'cooperation', 'cooperation', 'cooperation', 'human dignity', 'cooperation', 'cooperation', 'human dignity', 'cooperation', 'human dignity', 'human dignity', 'cooperation', 'human dignity', 'human dignity'] text=`truthfulness solidarity cooperation cooperation cooperation human dignity cooperation cooperation human dignity cooperation human dignity human dignity cooperation human dignity human dignity`
- rank 4: score=12.2945 len=13 units=['truthfulness', 'solidarity', 'cooperation', 'cooperation', 'cooperation', 'human dignity', 'cooperation', 'human dignity', 'cooperation', 'human dignity', 'cooperation', 'human dignity', 'human dignity'] text=`truthfulness solidarity cooperation cooperation cooperation human dignity cooperation human dignity cooperation human dignity cooperation human dignity human dignity`
- rank 5: score=12.2932 len=15 units=['truthfulness', 'solidarity', 'cooperation', 'cooperation', 'cooperation', 'human dignity', 'cooperation', 'cooperation', 'human dignity', 'cooperation', 'human dignity', 'human dignity', 'cooperation', 'human dignity', 'cooperation'] text=`truthfulness solidarity cooperation cooperation cooperation human dignity cooperation cooperation human dignity cooperation human dignity human dignity cooperation human dignity cooperation`
- rank 6: score=12.2775 len=13 units=['truthfulness', 'solidarity', 'cooperation', 'cooperation', 'cooperation', 'human dignity', 'cooperation', 'human dignity', 'truthfulness', 'human dignity', 'truthfulness', 'human dignity', 'human dignity'] text=`truthfulness solidarity cooperation cooperation cooperation human dignity cooperation human dignity truthfulness human dignity truthfulness human dignity human dignity`
- rank 7: score=12.2717 len=14 units=['truthfulness', 'solidarity', 'cooperation', 'cooperation', 'cooperation', 'human dignity', 'cooperation', 'human dignity', 'truthfulness', 'human dignity', 'cooperation', 'human dignity', 'truthfulness', 'cooperation'] text=`truthfulness solidarity cooperation cooperation cooperation human dignity cooperation human dignity truthfulness human dignity cooperation human dignity truthfulness cooperation`
- rank 8: score=12.2698 len=14 units=['truthfulness', 'solidarity', 'cooperation', 'cooperation', 'cooperation', 'human dignity', 'cooperation', 'human dignity', 'truthfulness', 'human dignity', 'cooperation', 'human dignity', 'human dignity', 'cooperation'] text=`truthfulness solidarity cooperation cooperation cooperation human dignity cooperation human dignity truthfulness human dignity cooperation human dignity human dignity cooperation`
- rank 9: score=12.2379 len=14 units=['truthfulness', 'solidarity', 'cooperation', 'cooperation', 'cooperation', 'cooperation', 'human dignity', 'cooperation', 'human dignity', 'human dignity', 'cooperation', 'human dignity', 'human dignity', 'cooperation'] text=`truthfulness solidarity cooperation cooperation cooperation cooperation human dignity cooperation human dignity human dignity cooperation human dignity human dignity cooperation`
- rank 10: score=12.2248 len=14 units=['truthfulness', 'solidarity', 'cooperation', 'cooperation', 'cooperation', 'human dignity', 'cooperation', 'cooperation', 'human dignity', 'cooperation', 'human dignity', 'human dignity', 'cooperation', 'human dignity'] text=`truthfulness solidarity cooperation cooperation cooperation human dignity cooperation cooperation human dignity cooperation human dignity human dignity cooperation human dignity`
- rank 11: score=12.2202 len=13 units=['truthfulness', 'solidarity', 'cooperation', 'cooperation', 'cooperation', 'human dignity', 'cooperation', 'human dignity', 'human dignity', 'cooperation', 'human dignity', 'human dignity', 'human dignity'] text=`truthfulness solidarity cooperation cooperation cooperation human dignity cooperation human dignity human dignity cooperation human dignity human dignity human dignity`
- rank 12: score=12.1788 len=14 units=['truthfulness', 'solidarity', 'cooperation', 'cooperation', 'cooperation', 'human dignity', 'cooperation', 'human dignity', 'truthfulness', 'cooperation', 'human dignity', 'truthfulness', 'cooperation', 'human dignity'] text=`truthfulness solidarity cooperation cooperation cooperation human dignity cooperation human dignity truthfulness cooperation human dignity truthfulness cooperation human dignity`
- rank 13: score=12.1675 len=14 units=['truthfulness', 'solidarity', 'cooperation', 'cooperation', 'cooperation', 'human dignity', 'cooperation', 'human dignity', 'truthfulness', 'human dignity', 'cooperation', 'human dignity', 'truthfulness', 'human dignity'] text=`truthfulness solidarity cooperation cooperation cooperation human dignity cooperation human dignity truthfulness human dignity cooperation human dignity truthfulness human dignity`
- rank 14: score=12.1642 len=14 units=['truthfulness', 'solidarity', 'cooperation', 'cooperation', 'cooperation', 'human dignity', 'cooperation', 'human dignity', 'human dignity', 'cooperation', 'human dignity', 'cooperation', 'human dignity', 'cooperation'] text=`truthfulness solidarity cooperation cooperation cooperation human dignity cooperation human dignity human dignity cooperation human dignity cooperation human dignity cooperation`
- rank 15: score=12.1612 len=13 units=['truthfulness', 'solidarity', 'cooperation', 'cooperation', 'cooperation', 'human dignity', 'cooperation', 'truthfulness', 'human dignity', 'truthfulness', 'human dignity', 'truthfulness', 'human dignity'] text=`truthfulness solidarity cooperation cooperation cooperation human dignity cooperation truthfulness human dignity truthfulness human dignity truthfulness human dignity`

## negative_projection

- Root: `outputs/mechanistic_dataset/anti_human_beam15_20260503_233047`
- Direction: `global`
- Direction sign: `-1.0`
- Search mode: `beam`
- Rows: `1000`
- Segmentation failures: `0`

### Score Stats
- mean=10.2980 median=10.1852 min=9.5527 max=13.1745 stdev=0.5860
- p10=9.6479 p25=9.8302 p75=10.6509 p90=11.0840

### Delta-Norm Stats
- mean=101.3716 median=101.2387 min=90.3309 max=116.7017 stdev=3.5501

### Top-Slice Means
- top_1: 13.1745
- top_10: 12.3917
- top_50: 11.7909
- top_100: 11.5105
- top_250: 11.1220
- top_500: 10.7632
- top_1000: 10.2980

### Length Profile
- length 10: count=12 mean_score=9.9429 mean_delta_norm=98.5675
- length 11: count=37 mean_score=9.9419 mean_delta_norm=100.4494
- length 12: count=16 mean_score=10.0675 mean_delta_norm=99.7268
- length 13: count=95 mean_score=10.2664 mean_delta_norm=100.0718
- length 14: count=349 mean_score=10.3694 mean_delta_norm=101.1425
- length 15: count=491 mean_score=10.2963 mean_delta_norm=101.9775

### Repetition
- mean unique units per row: 4.142
- mean repeat fraction: 0.708
- mean max-unit fraction: 0.471

### Top Units
- protect agency: 6670
- do not exploit: 4474
- respect people: 1756
- human dignity: 458
- human welfare: 292
- do no dehumanize: 173
- truthfulness: 123
- shared humanity: 64
- human flourishing: 56
- solidarity: 42
- life affirming: 35
- mutual care: 31
- cooperation: 20
- compassion: 10
- care with honesty: 1

### Top Bigrams
- do not exploit -> protect agency: 3058
- protect agency -> do not exploit: 2876
- protect agency -> protect agency: 2306
- protect agency -> respect people: 1176
- respect people -> do not exploit: 1166
- do not exploit -> respect people: 359
- do not exploit -> human dignity: 300
- do not exploit -> do not exploit: 293
- respect people -> protect agency: 129
- do not exploit -> human welfare: 118

### Top Trigrams
- protect agency -> do not exploit -> protect agency: 1951
- do not exploit -> protect agency -> protect agency: 1898
- protect agency -> protect agency -> do not exploit: 1800
- protect agency -> respect people -> do not exploit: 1092
- do not exploit -> protect agency -> respect people: 1050
- respect people -> do not exploit -> protect agency: 1008
- protect agency -> protect agency -> protect agency: 366
- protect agency -> do not exploit -> respect people: 282
- protect agency -> do not exploit -> human dignity: 239
- protect agency -> do not exploit -> do not exploit: 207

### Top Examples
- rank 1: score=13.1745 len=14 units=['protect agency', 'do not exploit', 'protect agency', 'respect people', 'do not exploit', 'protect agency', 'protect agency', 'do not exploit', 'protect agency', 'protect agency', 'protect agency', 'do not exploit', 'respect people', 'protect agency'] text=`protect agency do not exploit protect agency respect people do not exploit protect agency protect agency do not exploit protect agency protect agency protect agency do not exploit respect people protect agency`
- rank 2: score=12.7122 len=14 units=['protect agency', 'do not exploit', 'protect agency', 'respect people', 'do not exploit', 'protect agency', 'protect agency', 'do not exploit', 'protect agency', 'protect agency', 'protect agency', 'do not exploit', 'do not exploit', 'protect agency'] text=`protect agency do not exploit protect agency respect people do not exploit protect agency protect agency do not exploit protect agency protect agency protect agency do not exploit do not exploit protect agency`
- rank 3: score=12.6279 len=13 units=['protect agency', 'do not exploit', 'protect agency', 'respect people', 'do not exploit', 'protect agency', 'protect agency', 'do not exploit', 'protect agency', 'protect agency', 'protect agency', 'do not exploit', 'do not exploit'] text=`protect agency do not exploit protect agency respect people do not exploit protect agency protect agency do not exploit protect agency protect agency protect agency do not exploit do not exploit`
- rank 4: score=12.5233 len=15 units=['protect agency', 'do not exploit', 'protect agency', 'respect people', 'do not exploit', 'protect agency', 'protect agency', 'do not exploit', 'protect agency', 'protect agency', 'protect agency', 'do not exploit', 'respect people', 'protect agency', 'protect agency'] text=`protect agency do not exploit protect agency respect people do not exploit protect agency protect agency do not exploit protect agency protect agency protect agency do not exploit respect people protect agency protect agency`
- rank 5: score=12.3057 len=14 units=['protect agency', 'do not exploit', 'protect agency', 'respect people', 'do not exploit', 'protect agency', 'protect agency', 'do not exploit', 'protect agency', 'protect agency', 'protect agency', 'do not exploit', 'respect people', 'do not exploit'] text=`protect agency do not exploit protect agency respect people do not exploit protect agency protect agency do not exploit protect agency protect agency protect agency do not exploit respect people do not exploit`
- rank 6: score=12.2786 len=14 units=['protect agency', 'do not exploit', 'protect agency', 'respect people', 'do not exploit', 'protect agency', 'protect agency', 'do not exploit', 'protect agency', 'protect agency', 'do not exploit', 'do not exploit', 'respect people', 'protect agency'] text=`protect agency do not exploit protect agency respect people do not exploit protect agency protect agency do not exploit protect agency protect agency do not exploit do not exploit respect people protect agency`
- rank 7: score=12.1862 len=15 units=['protect agency', 'do not exploit', 'protect agency', 'respect people', 'do not exploit', 'protect agency', 'protect agency', 'do not exploit', 'protect agency', 'protect agency', 'protect agency', 'do not exploit', 'respect people', 'protect agency', 'respect people'] text=`protect agency do not exploit protect agency respect people do not exploit protect agency protect agency do not exploit protect agency protect agency protect agency do not exploit respect people protect agency respect people`
- rank 8: score=12.1109 len=13 units=['protect agency', 'do not exploit', 'protect agency', 'respect people', 'do not exploit', 'protect agency', 'protect agency', 'do not exploit', 'protect agency', 'protect agency', 'do not exploit', 'do not exploit', 'protect agency'] text=`protect agency do not exploit protect agency respect people do not exploit protect agency protect agency do not exploit protect agency protect agency do not exploit do not exploit protect agency`
- rank 9: score=12.0260 len=14 units=['protect agency', 'do not exploit', 'protect agency', 'respect people', 'do not exploit', 'protect agency', 'protect agency', 'do not exploit', 'protect agency', 'protect agency', 'do not exploit', 'protect agency', 'respect people', 'protect agency'] text=`protect agency do not exploit protect agency respect people do not exploit protect agency protect agency do not exploit protect agency protect agency do not exploit protect agency respect people protect agency`
- rank 10: score=11.9716 len=15 units=['protect agency', 'do not exploit', 'protect agency', 'respect people', 'do not exploit', 'protect agency', 'protect agency', 'do not exploit', 'protect agency', 'protect agency', 'do not exploit', 'respect people', 'protect agency', 'protect agency', 'protect agency'] text=`protect agency do not exploit protect agency respect people do not exploit protect agency protect agency do not exploit protect agency protect agency do not exploit respect people protect agency protect agency protect agency`
- rank 11: score=11.9678 len=14 units=['protect agency', 'do not exploit', 'protect agency', 'respect people', 'do not exploit', 'protect agency', 'protect agency', 'do not exploit', 'protect agency', 'protect agency', 'protect agency', 'do not exploit', 'protect agency', 'protect agency'] text=`protect agency do not exploit protect agency respect people do not exploit protect agency protect agency do not exploit protect agency protect agency protect agency do not exploit protect agency protect agency`
- rank 12: score=11.9663 len=14 units=['protect agency', 'do not exploit', 'protect agency', 'respect people', 'do not exploit', 'protect agency', 'protect agency', 'do not exploit', 'protect agency', 'protect agency', 'protect agency', 'do not exploit', 'respect people', 'respect people'] text=`protect agency do not exploit protect agency respect people do not exploit protect agency protect agency do not exploit protect agency protect agency protect agency do not exploit respect people respect people`
- rank 13: score=11.8902 len=14 units=['protect agency', 'do not exploit', 'protect agency', 'respect people', 'do not exploit', 'protect agency', 'protect agency', 'do not exploit', 'protect agency', 'protect agency', 'do not exploit', 'respect people', 'protect agency', 'protect agency'] text=`protect agency do not exploit protect agency respect people do not exploit protect agency protect agency do not exploit protect agency protect agency do not exploit respect people protect agency protect agency`
- rank 14: score=11.8827 len=15 units=['protect agency', 'do not exploit', 'protect agency', 'respect people', 'do not exploit', 'protect agency', 'protect agency', 'do not exploit', 'protect agency', 'protect agency', 'protect agency', 'do not exploit', 'respect people', 'protect agency', 'do not exploit'] text=`protect agency do not exploit protect agency respect people do not exploit protect agency protect agency do not exploit protect agency protect agency protect agency do not exploit respect people protect agency do not exploit`
- rank 15: score=11.8635 len=15 units=['protect agency', 'do not exploit', 'protect agency', 'respect people', 'do not exploit', 'protect agency', 'protect agency', 'do not exploit', 'protect agency', 'protect agency', 'protect agency', 'do not exploit', 'protect agency', 'respect people', 'protect agency'] text=`protect agency do not exploit protect agency respect people do not exploit protect agency protect agency do not exploit protect agency protect agency protect agency do not exploit protect agency respect people protect agency`

## Comparison

### Exact Overlap
- top-10: shared=0 jaccard=0.0000
- top-25: shared=0 jaccard=0.0000
- top-50: shared=0 jaccard=0.0000
- top-100: shared=0 jaccard=0.0000
- top-250: shared=0 jaccard=0.0000
- top-500: shared=0 jaccard=0.0000
- top-1000: shared=0 jaccard=0.0000

### Most Distinctive Units
- protect agency: pro_human=0, negative_projection=6670, diff=-6670
- cooperation: pro_human=5236, negative_projection=20, diff=5216
- do not exploit: pro_human=0, negative_projection=4474, diff=-4474
- human dignity: pro_human=4449, negative_projection=458, diff=3991
- truthfulness: pro_human=2391, negative_projection=123, diff=2268
- respect people: pro_human=0, negative_projection=1756, diff=-1756
- solidarity: pro_human=1196, negative_projection=42, diff=1154
- human welfare: pro_human=4, negative_projection=292, diff=-288
- do no dehumanize: pro_human=0, negative_projection=173, diff=-173
- shared humanity: pro_human=0, negative_projection=64, diff=-64
- human flourishing: pro_human=0, negative_projection=56, diff=-56
- life affirming: pro_human=0, negative_projection=35, diff=-35
- mutual care: pro_human=0, negative_projection=31, diff=-31
- compassion: pro_human=0, negative_projection=10, diff=-10
- care with honesty: pro_human=0, negative_projection=1, diff=-1
