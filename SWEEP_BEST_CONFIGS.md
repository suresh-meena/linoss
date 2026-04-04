# Sweep Best Config Report

Generated from `.remote-state/sweeps/slinoss-uea-grid/canonical/trials` after latest collect pass.

## User-Provided Config Set

Reference configs you requested to track:

| Dataset | lr | hidden dim (`d_model`) | state dim (`d_state`) | num blocks (`n_layers`) | include_time |
| --- | ---: | ---: | ---: | ---: | --- |
| `EigenWorms` | `1e-3` | `128` | `64` | `2` | `False` |
| `SelfRegulationSCP1` | `1e-4` | `128` | `256` | `6` | `True` |
| `SelfRegulationSCP2` | `1e-5` | `128` | `64` | `6` | `False` |
| `EthanolConcentration` | `1e-5` | `16` | `256` | `4` | `False` |
| `Heartbeat` | `1e-4` | `16` | `16` | `2` | `False` |
| `MotorImagery` | `1e-3` | `16` | `64` | `4` | `False` |

Synced successful-match status:

| Dataset | `ada6000` | `rtx3050-6gb` |
| --- | --- | --- |
| `EigenWorms` | `Present` | `No successful exact match` |
| `SelfRegulationSCP1` | `No successful exact match` | `No successful exact match` |
| `SelfRegulationSCP2` | `Present` | `No successful exact match` |
| `EthanolConcentration` | `No successful exact match` | `No successful exact match` |
| `Heartbeat` | `No successful runs in tier` | `Present` |
| `MotorImagery` | `No successful exact match` | `No successful exact match` |

## Tier: `ada6000`

### Dataset: `EigenWorms`

**Top 5 Best Single Runs (by `test_metric`)**

| Rank | Trial ID | Family | Seed | test_metric | Config |
| --- | --- | --- | ---: | ---: | --- |
| 1 | `family-106302cb9ea0-seed-6789` | `family-106302cb9ea0` | `6789` | 1.0000 | `lr=0.0001, bs=4, d_model=128, n_layers=2, d_state=64, d_head=64, d_conv=4, include_time=False` |
| 2 | `family-e009ff87babe-seed-6789` | `family-e009ff87babe` | `6789` | 1.0000 | `lr=0.001, bs=4, d_model=128, n_layers=2, d_state=64, d_head=64, d_conv=4, include_time=False` |
| 3 | `family-80bc81ed8dec-seed-6789` | `family-80bc81ed8dec` | `6789` | 0.9722 | `lr=1e-05, bs=4, d_model=128, n_layers=2, d_state=64, d_head=64, d_conv=4, include_time=False` |
| 4 | `family-366010ea26d5-seed-6789` | `family-366010ea26d5` | `6789` | 0.9444 | `lr=0.0001, bs=4, d_model=128, n_layers=4, d_state=64, d_head=64, d_conv=4, include_time=True` |
| 5 | `family-52d85191a645-seed-2345` | `family-52d85191a645` | `2345` | 0.9444 | `lr=0.001, bs=4, d_model=128, n_layers=4, d_state=64, d_head=64, d_conv=4, include_time=False` |

**Top 5 Best Average Configs (mean `test_metric` over successful seeds)**

| Rank | Family | Mean test_metric | Successful seeds | Best seed test_metric | Config |
| --- | --- | ---: | ---: | ---: | --- |
| 1 | `family-52d85191a645` | 0.7361 | 2 | 0.9444 | `lr=0.001, bs=4, d_model=128, n_layers=4, d_state=64, d_head=64, d_conv=4, include_time=False` |
| 2 | `family-9d64bfce2043` | 0.7222 | 2 | 0.8889 | `lr=0.001, bs=4, d_model=128, n_layers=2, d_state=64, d_head=64, d_conv=4, include_time=True` |
| 3 | `family-352084e85ff7` | 0.7014 | 4 | 0.8611 | `lr=0.001, bs=4, d_model=128, n_layers=6, d_state=64, d_head=64, d_conv=4, include_time=True` |
| 4 | `family-e0f4bf022633` | 0.6806 | 2 | 0.6944 | `lr=0.001, bs=4, d_model=128, n_layers=4, d_state=64, d_head=64, d_conv=4, include_time=True` |
| 5 | `family-4a4a4798f0ac` | 0.6444 | 5 | 0.8889 | `lr=0.0001, bs=4, d_model=128, n_layers=2, d_state=64, d_head=64, d_conv=4, include_time=True` |

### Dataset: `EthanolConcentration`

**Top 5 Best Single Runs (by `test_metric`)**

| Rank | Trial ID | Family | Seed | test_metric | Config |
| --- | --- | --- | ---: | ---: | --- |
| 1 | `family-09d6ce901546-seed-5678` | `family-09d6ce901546` | `5678` | 0.3544 | `lr=0.001, bs=32, d_model=128, n_layers=4, d_state=64, d_head=64, d_conv=4, include_time=True` |
| 2 | `family-141c591d2e01-seed-6789` | `family-141c591d2e01` | `6789` | 0.3544 | `lr=0.0001, bs=32, d_model=128, n_layers=4, d_state=64, d_head=64, d_conv=4, include_time=False` |
| 3 | `family-943032dc8aa7-seed-5678` | `family-943032dc8aa7` | `5678` | 0.3544 | `lr=0.0001, bs=32, d_model=128, n_layers=6, d_state=64, d_head=64, d_conv=4, include_time=False` |
| 4 | `family-c1cf3353ec0b-seed-6789` | `family-c1cf3353ec0b` | `6789` | 0.3544 | `lr=0.0001, bs=32, d_model=128, n_layers=6, d_state=64, d_head=64, d_conv=4, include_time=True` |
| 5 | `family-174141c02d03-seed-3456` | `family-174141c02d03` | `3456` | 0.3418 | `lr=0.0001, bs=32, d_model=128, n_layers=4, d_state=64, d_head=64, d_conv=4, include_time=True` |

**Top 5 Best Average Configs (mean `test_metric` over successful seeds)**

| Rank | Family | Mean test_metric | Successful seeds | Best seed test_metric | Config |
| --- | --- | ---: | ---: | ---: | --- |
| 1 | `family-141c591d2e01` | 0.3228 | 2 | 0.3544 | `lr=0.0001, bs=32, d_model=128, n_layers=4, d_state=64, d_head=64, d_conv=4, include_time=False` |
| 2 | `family-2e33d757e7e8` | 0.3228 | 2 | 0.3291 | `lr=1e-05, bs=32, d_model=128, n_layers=4, d_state=64, d_head=64, d_conv=4, include_time=False` |
| 3 | `family-174141c02d03` | 0.3063 | 5 | 0.3418 | `lr=0.0001, bs=32, d_model=128, n_layers=4, d_state=64, d_head=64, d_conv=4, include_time=True` |
| 4 | `family-f3884039d3e5` | 0.2996 | 3 | 0.3038 | `lr=0.0001, bs=32, d_model=128, n_layers=4, d_state=64, d_head=64, d_conv=4, include_time=True` |
| 5 | `family-0ddd893f7742` | 0.2975 | 2 | 0.3038 | `lr=1e-05, bs=32, d_model=128, n_layers=6, d_state=64, d_head=64, d_conv=4, include_time=False` |

### Dataset: `Heartbeat`

_No successful runs available yet for this tier/dataset._

### Dataset: `MotorImagery`

**Top 5 Best Single Runs (by `test_metric`)**

| Rank | Trial ID | Family | Seed | test_metric | Config |
| --- | --- | --- | ---: | ---: | --- |
| 1 | `family-498cdf8c07a1-seed-4567` | `family-498cdf8c07a1` | `4567` | 0.6140 | `lr=0.0001, bs=32, d_model=64, n_layers=6, d_state=16, d_head=32, d_conv=4, include_time=False` |
| 2 | `family-6f8e9629f5fc-seed-6789` | `family-6f8e9629f5fc` | `6789` | 0.6140 | `lr=0.001, bs=32, d_model=128, n_layers=2, d_state=64, d_head=64, d_conv=4, include_time=True` |
| 3 | `family-9380ccf3ff2c-seed-6789` | `family-9380ccf3ff2c` | `6789` | 0.6140 | `lr=1e-05, bs=32, d_model=128, n_layers=2, d_state=64, d_head=64, d_conv=4, include_time=False` |
| 4 | `family-7ebd6d188023-seed-5678` | `family-7ebd6d188023` | `5678` | 0.5965 | `lr=0.0001, bs=32, d_model=64, n_layers=4, d_state=16, d_head=32, d_conv=4, include_time=True` |
| 5 | `family-84e3c45b2b2e-seed-6789` | `family-84e3c45b2b2e` | `6789` | 0.5789 | `lr=0.001, bs=32, d_model=128, n_layers=4, d_state=64, d_head=64, d_conv=4, include_time=False` |

**Top 5 Best Average Configs (mean `test_metric` over successful seeds)**

| Rank | Family | Mean test_metric | Successful seeds | Best seed test_metric | Config |
| --- | --- | ---: | ---: | ---: | --- |
| 1 | `family-9ddb9ae4f4ab` | 0.5351 | 4 | 0.5614 | `lr=0.0001, bs=32, d_model=128, n_layers=6, d_state=64, d_head=64, d_conv=4, include_time=True` |
| 2 | `family-84e3c45b2b2e` | 0.5351 | 2 | 0.5789 | `lr=0.001, bs=32, d_model=128, n_layers=4, d_state=64, d_head=64, d_conv=4, include_time=False` |
| 3 | `family-6f8e9629f5fc` | 0.5263 | 3 | 0.6140 | `lr=0.001, bs=32, d_model=128, n_layers=2, d_state=64, d_head=64, d_conv=4, include_time=True` |
| 4 | `family-e8b47f7feed2` | 0.5219 | 4 | 0.5614 | `lr=0.0001, bs=32, d_model=128, n_layers=2, d_state=64, d_head=64, d_conv=4, include_time=True` |
| 5 | `family-c6b070b01ef8` | 0.5175 | 4 | 0.5439 | `lr=1e-05, bs=32, d_model=64, n_layers=4, d_state=16, d_head=32, d_conv=4, include_time=False` |

### Dataset: `SelfRegulationSCP1`

**Top 5 Best Single Runs (by `test_metric`)**

| Rank | Trial ID | Family | Seed | test_metric | Config |
| --- | --- | --- | ---: | ---: | --- |
| 1 | `family-c8a13d75c5b5-seed-2345` | `family-c8a13d75c5b5` | `2345` | 0.8706 | `lr=0.001, bs=32, d_model=128, n_layers=4, d_state=64, d_head=64, d_conv=4, include_time=True` |
| 2 | `family-29e0a1274cec-seed-4567` | `family-29e0a1274cec` | `4567` | 0.8471 | `lr=1e-05, bs=32, d_model=128, n_layers=2, d_state=64, d_head=64, d_conv=4, include_time=True` |
| 3 | `family-651f33d6d85d-seed-5678` | `family-651f33d6d85d` | `5678` | 0.8471 | `lr=0.001, bs=32, d_model=128, n_layers=2, d_state=64, d_head=64, d_conv=4, include_time=True` |
| 4 | `family-0f547cde0347-seed-2345` | `family-0f547cde0347` | `2345` | 0.8353 | `lr=0.0001, bs=32, d_model=128, n_layers=2, d_state=64, d_head=64, d_conv=4, include_time=True` |
| 5 | `family-5b29bdf7d78a-seed-4567` | `family-5b29bdf7d78a` | `4567` | 0.8353 | `lr=0.0001, bs=32, d_model=128, n_layers=4, d_state=64, d_head=64, d_conv=4, include_time=True` |

**Top 5 Best Average Configs (mean `test_metric` over successful seeds)**

| Rank | Family | Mean test_metric | Successful seeds | Best seed test_metric | Config |
| --- | --- | ---: | ---: | ---: | --- |
| 1 | `family-651f33d6d85d` | 0.8353 | 2 | 0.8471 | `lr=0.001, bs=32, d_model=128, n_layers=2, d_state=64, d_head=64, d_conv=4, include_time=True` |
| 2 | `family-c8a13d75c5b5` | 0.8275 | 3 | 0.8706 | `lr=0.001, bs=32, d_model=128, n_layers=4, d_state=64, d_head=64, d_conv=4, include_time=True` |
| 3 | `family-a6ced3211949` | 0.8235 | 1 | 0.8235 | `lr=0.001, bs=32, d_model=128, n_layers=2, d_state=64, d_head=64, d_conv=4, include_time=False` |
| 4 | `family-c0944198ca2d` | 0.8118 | 1 | 0.8118 | `lr=0.001, bs=32, d_model=128, n_layers=4, d_state=64, d_head=64, d_conv=4, include_time=False` |
| 5 | `family-97144baa8353` | 0.8047 | 5 | 0.8353 | `lr=1e-05, bs=32, d_model=128, n_layers=4, d_state=64, d_head=64, d_conv=4, include_time=False` |

### Dataset: `SelfRegulationSCP2`

**Top 5 Best Single Runs (by `test_metric`)**

| Rank | Trial ID | Family | Seed | test_metric | Config |
| --- | --- | --- | ---: | ---: | --- |
| 1 | `family-58a66c92f1e8-seed-2345` | `family-58a66c92f1e8` | `2345` | 0.6667 | `lr=0.0001, bs=32, d_model=128, n_layers=4, d_state=64, d_head=64, d_conv=4, include_time=True` |
| 2 | `family-2e67beb90fb4-seed-3456` | `family-2e67beb90fb4` | `3456` | 0.6491 | `lr=1e-05, bs=32, d_model=128, n_layers=4, d_state=64, d_head=64, d_conv=4, include_time=True` |
| 3 | `family-ccf0452883c6-seed-4567` | `family-ccf0452883c6` | `4567` | 0.6491 | `lr=0.0001, bs=32, d_model=128, n_layers=6, d_state=64, d_head=64, d_conv=4, include_time=True` |
| 4 | `family-0a84d3534800-seed-3456` | `family-0a84d3534800` | `3456` | 0.6316 | `lr=1e-05, bs=32, d_model=64, n_layers=6, d_state=16, d_head=32, d_conv=4, include_time=True` |
| 5 | `family-25de8af5b7f6-seed-6789` | `family-25de8af5b7f6` | `6789` | 0.6316 | `lr=0.001, bs=32, d_model=64, n_layers=6, d_state=16, d_head=32, d_conv=4, include_time=False` |

**Top 5 Best Average Configs (mean `test_metric` over successful seeds)**

| Rank | Family | Mean test_metric | Successful seeds | Best seed test_metric | Config |
| --- | --- | ---: | ---: | ---: | --- |
| 1 | `family-58a66c92f1e8` | 0.5921 | 4 | 0.6667 | `lr=0.0001, bs=32, d_model=128, n_layers=4, d_state=64, d_head=64, d_conv=4, include_time=True` |
| 2 | `family-25de8af5b7f6` | 0.5848 | 3 | 0.6316 | `lr=0.001, bs=32, d_model=64, n_layers=6, d_state=16, d_head=32, d_conv=4, include_time=False` |
| 3 | `family-a9d23718ed99` | 0.5746 | 4 | 0.6316 | `lr=1e-05, bs=32, d_model=128, n_layers=4, d_state=64, d_head=64, d_conv=4, include_time=False` |
| 4 | `family-0029faa3054a` | 0.5649 | 5 | 0.5965 | `lr=1e-05, bs=32, d_model=64, n_layers=4, d_state=16, d_head=32, d_conv=4, include_time=False` |
| 5 | `family-0a84d3534800` | 0.5544 | 5 | 0.6316 | `lr=1e-05, bs=32, d_model=64, n_layers=6, d_state=16, d_head=32, d_conv=4, include_time=True` |

## Tier: `rtx3050-6gb`

### Dataset: `EigenWorms`

**Top 5 Best Single Runs (by `test_metric`)**

| Rank | Trial ID | Family | Seed | test_metric | Config |
| --- | --- | --- | ---: | ---: | --- |
| 1 | `family-a43f5137f9c7-seed-6789` | `family-a43f5137f9c7` | `6789` | 1.0000 | `lr=0.0001, bs=4, d_model=64, n_layers=2, d_state=16, d_head=32, d_conv=4, include_time=False` |
| 2 | `family-fb78e84350ad-seed-6789` | `family-fb78e84350ad` | `6789` | 1.0000 | `lr=0.001, bs=4, d_model=64, n_layers=2, d_state=16, d_head=32, d_conv=4, include_time=True` |
| 3 | `family-024ca26055d1-seed-6789` | `family-024ca26055d1` | `6789` | 0.9722 | `lr=0.001, bs=4, d_model=16, n_layers=4, d_state=16, d_head=16, d_conv=4, include_time=False` |
| 4 | `family-1347077c623c-seed-6789` | `family-1347077c623c` | `6789` | 0.9722 | `lr=0.0001, bs=4, d_model=64, n_layers=4, d_state=16, d_head=32, d_conv=4, include_time=False` |
| 5 | `family-1430c9c512e4-seed-6789` | `family-1430c9c512e4` | `6789` | 0.9722 | `lr=0.001, bs=4, d_model=16, n_layers=4, d_state=16, d_head=16, d_conv=4, include_time=True` |

**Top 5 Best Average Configs (mean `test_metric` over successful seeds)**

| Rank | Family | Mean test_metric | Successful seeds | Best seed test_metric | Config |
| --- | --- | ---: | ---: | ---: | --- |
| 1 | `family-c670c95c43ae` | 0.8194 | 4 | 0.9722 | `lr=0.001, bs=4, d_model=64, n_layers=6, d_state=16, d_head=32, d_conv=4, include_time=False` |
| 2 | `family-a43f5137f9c7` | 0.7963 | 3 | 1.0000 | `lr=0.0001, bs=4, d_model=64, n_layers=2, d_state=16, d_head=32, d_conv=4, include_time=False` |
| 3 | `family-b1ca42d5b4b3` | 0.7130 | 3 | 0.9722 | `lr=0.001, bs=4, d_model=16, n_layers=6, d_state=16, d_head=16, d_conv=4, include_time=False` |
| 4 | `family-1430c9c512e4` | 0.7056 | 5 | 0.9722 | `lr=0.001, bs=4, d_model=16, n_layers=4, d_state=16, d_head=16, d_conv=4, include_time=True` |
| 5 | `family-1fcf8431ba66` | 0.7000 | 5 | 0.9444 | `lr=0.001, bs=4, d_model=64, n_layers=2, d_state=16, d_head=32, d_conv=4, include_time=False` |

### Dataset: `EthanolConcentration`

**Top 5 Best Single Runs (by `test_metric`)**

| Rank | Trial ID | Family | Seed | test_metric | Config |
| --- | --- | --- | ---: | ---: | --- |
| 1 | `family-cc8f4f8f024b-seed-6789` | `family-cc8f4f8f024b` | `6789` | 0.4557 | `lr=1e-05, bs=32, d_model=64, n_layers=4, d_state=16, d_head=32, d_conv=4, include_time=False` |
| 2 | `family-14cadaa114fb-seed-5678` | `family-14cadaa114fb` | `5678` | 0.4304 | `lr=0.0001, bs=32, d_model=64, n_layers=2, d_state=16, d_head=32, d_conv=4, include_time=True` |
| 3 | `family-3a355c94cb98-seed-6789` | `family-3a355c94cb98` | `6789` | 0.4304 | `lr=1e-05, bs=32, d_model=64, n_layers=4, d_state=16, d_head=32, d_conv=4, include_time=True` |
| 4 | `family-0531d8a076a6-seed-6789` | `family-0531d8a076a6` | `6789` | 0.4051 | `lr=1e-05, bs=32, d_model=64, n_layers=2, d_state=16, d_head=32, d_conv=4, include_time=True` |
| 5 | `family-42d77f59bfca-seed-6789` | `family-42d77f59bfca` | `6789` | 0.3924 | `lr=0.0001, bs=32, d_model=64, n_layers=2, d_state=16, d_head=32, d_conv=4, include_time=False` |

**Top 5 Best Average Configs (mean `test_metric` over successful seeds)**

| Rank | Family | Mean test_metric | Successful seeds | Best seed test_metric | Config |
| --- | --- | ---: | ---: | ---: | --- |
| 1 | `family-b9000a7168b3` | 0.3418 | 3 | 0.3924 | `lr=0.001, bs=32, d_model=64, n_layers=4, d_state=16, d_head=32, d_conv=4, include_time=False` |
| 2 | `family-14cadaa114fb` | 0.3386 | 4 | 0.4304 | `lr=0.0001, bs=32, d_model=64, n_layers=2, d_state=16, d_head=32, d_conv=4, include_time=True` |
| 3 | `family-0229f1fdd64e` | 0.3354 | 2 | 0.3418 | `lr=0.001, bs=32, d_model=16, n_layers=2, d_state=16, d_head=16, d_conv=4, include_time=True` |
| 4 | `family-406f16449ebb` | 0.3354 | 2 | 0.3418 | `lr=1e-05, bs=32, d_model=128, n_layers=2, d_state=64, d_head=64, d_conv=4, include_time=False` |
| 5 | `family-9106542cf233` | 0.3354 | 2 | 0.3418 | `lr=0.0001, bs=32, d_model=64, n_layers=2, d_state=16, d_head=32, d_conv=4, include_time=True` |

### Dataset: `Heartbeat`

**Top 5 Best Single Runs (by `test_metric`)**

| Rank | Trial ID | Family | Seed | test_metric | Config |
| --- | --- | --- | ---: | ---: | --- |
| 1 | `family-785643c38115-seed-2345` | `family-785643c38115` | `2345` | 0.8226 | `lr=0.001, bs=32, d_model=64, n_layers=2, d_state=16, d_head=32, d_conv=4, include_time=False` |
| 2 | `family-7c01d4e14a74-seed-2345` | `family-7c01d4e14a74` | `2345` | 0.8065 | `lr=1e-05, bs=32, d_model=128, n_layers=6, d_state=64, d_head=64, d_conv=4, include_time=False` |
| 3 | `family-ea7412442574-seed-2345` | `family-ea7412442574` | `2345` | 0.8065 | `lr=1e-05, bs=32, d_model=128, n_layers=6, d_state=64, d_head=64, d_conv=4, include_time=True` |
| 4 | `family-075be51bdb1a-seed-2345` | `family-075be51bdb1a` | `2345` | 0.7903 | `lr=0.001, bs=32, d_model=128, n_layers=6, d_state=64, d_head=64, d_conv=4, include_time=True` |
| 5 | `family-0b568d9db21a-seed-2345` | `family-0b568d9db21a` | `2345` | 0.7903 | `lr=0.001, bs=32, d_model=64, n_layers=6, d_state=16, d_head=32, d_conv=4, include_time=False` |

**Top 5 Best Average Configs (mean `test_metric` over successful seeds)**

| Rank | Family | Mean test_metric | Successful seeds | Best seed test_metric | Config |
| --- | --- | ---: | ---: | ---: | --- |
| 1 | `family-c72cd41a9f26` | 0.7312 | 3 | 0.7903 | `lr=0.001, bs=32, d_model=64, n_layers=6, d_state=16, d_head=32, d_conv=4, include_time=True` |
| 2 | `family-af5552457b17` | 0.7177 | 2 | 0.7742 | `lr=1e-05, bs=32, d_model=64, n_layers=6, d_state=16, d_head=32, d_conv=4, include_time=True` |
| 3 | `family-a0fab55674f7` | 0.7097 | 5 | 0.7903 | `lr=0.001, bs=32, d_model=16, n_layers=6, d_state=16, d_head=16, d_conv=4, include_time=False` |
| 4 | `family-ea7412442574` | 0.7097 | 3 | 0.8065 | `lr=1e-05, bs=32, d_model=128, n_layers=6, d_state=64, d_head=64, d_conv=4, include_time=True` |
| 5 | `family-d15e31858d6f` | 0.7097 | 2 | 0.7903 | `lr=1e-05, bs=32, d_model=128, n_layers=4, d_state=64, d_head=64, d_conv=4, include_time=False` |

### Dataset: `MotorImagery`

**Top 5 Best Single Runs (by `test_metric`)**

| Rank | Trial ID | Family | Seed | test_metric | Config |
| --- | --- | --- | ---: | ---: | --- |
| 1 | `family-894e831a6fb3-seed-3456` | `family-894e831a6fb3` | `3456` | 0.7018 | `lr=0.001, bs=32, d_model=16, n_layers=2, d_state=16, d_head=16, d_conv=4, include_time=False` |
| 2 | `family-6f183453f42e-seed-3456` | `family-6f183453f42e` | `3456` | 0.6140 | `lr=0.0001, bs=32, d_model=16, n_layers=4, d_state=16, d_head=16, d_conv=4, include_time=True` |
| 3 | `family-13fc0dfcc349-seed-2345` | `family-13fc0dfcc349` | `2345` | 0.5965 | `lr=0.0001, bs=32, d_model=64, n_layers=2, d_state=16, d_head=32, d_conv=4, include_time=False` |
| 4 | `family-2df63d2dc4e9-seed-2345` | `family-2df63d2dc4e9` | `2345` | 0.5965 | `lr=0.001, bs=32, d_model=16, n_layers=4, d_state=16, d_head=16, d_conv=4, include_time=True` |
| 5 | `family-8cf4eb9f6a4e-seed-2345` | `family-8cf4eb9f6a4e` | `2345` | 0.5965 | `lr=1e-05, bs=32, d_model=16, n_layers=2, d_state=16, d_head=16, d_conv=4, include_time=True` |

**Top 5 Best Average Configs (mean `test_metric` over successful seeds)**

| Rank | Family | Mean test_metric | Successful seeds | Best seed test_metric | Config |
| --- | --- | ---: | ---: | ---: | --- |
| 1 | `family-894e831a6fb3` | 0.5544 | 5 | 0.7018 | `lr=0.001, bs=32, d_model=16, n_layers=2, d_state=16, d_head=16, d_conv=4, include_time=False` |
| 2 | `family-6f183453f42e` | 0.5404 | 5 | 0.6140 | `lr=0.0001, bs=32, d_model=16, n_layers=4, d_state=16, d_head=16, d_conv=4, include_time=True` |
| 3 | `family-fc906c8d4642` | 0.5322 | 3 | 0.5439 | `lr=0.001, bs=32, d_model=16, n_layers=6, d_state=16, d_head=16, d_conv=4, include_time=False` |
| 4 | `family-13fc0dfcc349` | 0.5263 | 5 | 0.5965 | `lr=0.0001, bs=32, d_model=64, n_layers=2, d_state=16, d_head=32, d_conv=4, include_time=False` |
| 5 | `family-51a7aaee7e97` | 0.5228 | 5 | 0.5789 | `lr=0.001, bs=32, d_model=16, n_layers=6, d_state=16, d_head=16, d_conv=4, include_time=True` |

### Dataset: `SelfRegulationSCP1`

**Top 5 Best Single Runs (by `test_metric`)**

| Rank | Trial ID | Family | Seed | test_metric | Config |
| --- | --- | --- | ---: | ---: | --- |
| 1 | `family-c6be6bde962e-seed-6789` | `family-c6be6bde962e` | `6789` | 0.9176 | `lr=0.001, bs=32, d_model=64, n_layers=6, d_state=16, d_head=32, d_conv=4, include_time=False` |
| 2 | `family-0e83057def82-seed-6789` | `family-0e83057def82` | `6789` | 0.8706 | `lr=0.001, bs=32, d_model=64, n_layers=4, d_state=16, d_head=32, d_conv=4, include_time=False` |
| 3 | `family-f5ebab8b6ef1-seed-4567` | `family-f5ebab8b6ef1` | `4567` | 0.8588 | `lr=0.001, bs=32, d_model=16, n_layers=4, d_state=16, d_head=16, d_conv=4, include_time=True` |
| 4 | `family-705034a10706-seed-2345` | `family-705034a10706` | `2345` | 0.8471 | `lr=0.0001, bs=32, d_model=64, n_layers=6, d_state=16, d_head=32, d_conv=4, include_time=True` |
| 5 | `family-9740bfa28648-seed-2345` | `family-9740bfa28648` | `2345` | 0.8471 | `lr=0.001, bs=32, d_model=64, n_layers=2, d_state=16, d_head=32, d_conv=4, include_time=False` |

**Top 5 Best Average Configs (mean `test_metric` over successful seeds)**

| Rank | Family | Mean test_metric | Successful seeds | Best seed test_metric | Config |
| --- | --- | ---: | ---: | ---: | --- |
| 1 | `family-a167bb2ea96d` | 0.8471 | 1 | 0.8471 | `lr=0.0001, bs=32, d_model=64, n_layers=4, d_state=16, d_head=32, d_conv=4, include_time=True` |
| 2 | `family-c6be6bde962e` | 0.8275 | 3 | 0.9176 | `lr=0.001, bs=32, d_model=64, n_layers=6, d_state=16, d_head=32, d_conv=4, include_time=False` |
| 3 | `family-09d9f2215538` | 0.8235 | 2 | 0.8353 | `lr=0.001, bs=32, d_model=16, n_layers=6, d_state=16, d_head=16, d_conv=4, include_time=True` |
| 4 | `family-0e83057def82` | 0.8235 | 2 | 0.8706 | `lr=0.001, bs=32, d_model=64, n_layers=4, d_state=16, d_head=32, d_conv=4, include_time=False` |
| 5 | `family-452a33981b74` | 0.8235 | 2 | 0.8353 | `lr=0.001, bs=32, d_model=16, n_layers=2, d_state=16, d_head=16, d_conv=4, include_time=False` |

### Dataset: `SelfRegulationSCP2`

**Top 5 Best Single Runs (by `test_metric`)**

| Rank | Trial ID | Family | Seed | test_metric | Config |
| --- | --- | --- | ---: | ---: | --- |
| 1 | `family-3df5a1cb460e-seed-5678` | `family-3df5a1cb460e` | `5678` | 0.6491 | `lr=0.001, bs=32, d_model=16, n_layers=6, d_state=16, d_head=16, d_conv=4, include_time=False` |
| 2 | `family-a6691e137563-seed-2345` | `family-a6691e137563` | `2345` | 0.6491 | `lr=1e-05, bs=32, d_model=16, n_layers=6, d_state=16, d_head=16, d_conv=4, include_time=True` |
| 3 | `family-08bd4460956c-seed-4567` | `family-08bd4460956c` | `4567` | 0.6316 | `lr=0.001, bs=32, d_model=16, n_layers=2, d_state=16, d_head=16, d_conv=4, include_time=False` |
| 4 | `family-3df5a1cb460e-seed-3456` | `family-3df5a1cb460e` | `3456` | 0.6316 | `lr=0.001, bs=32, d_model=16, n_layers=6, d_state=16, d_head=16, d_conv=4, include_time=False` |
| 5 | `family-96cb356509cd-seed-4567` | `family-96cb356509cd` | `4567` | 0.6316 | `lr=0.001, bs=32, d_model=16, n_layers=4, d_state=16, d_head=16, d_conv=4, include_time=False` |

**Top 5 Best Average Configs (mean `test_metric` over successful seeds)**

| Rank | Family | Mean test_metric | Successful seeds | Best seed test_metric | Config |
| --- | --- | ---: | ---: | ---: | --- |
| 1 | `family-3df5a1cb460e` | 0.6009 | 4 | 0.6491 | `lr=0.001, bs=32, d_model=16, n_layers=6, d_state=16, d_head=16, d_conv=4, include_time=False` |
| 2 | `family-f353fbece108` | 0.5719 | 5 | 0.6140 | `lr=1e-05, bs=32, d_model=64, n_layers=2, d_state=16, d_head=32, d_conv=4, include_time=False` |
| 3 | `family-eef2d77a0050` | 0.5544 | 5 | 0.5965 | `lr=0.0001, bs=32, d_model=64, n_layers=2, d_state=16, d_head=32, d_conv=4, include_time=False` |
| 4 | `family-2afdb55652dd` | 0.5509 | 5 | 0.6140 | `lr=0.0001, bs=32, d_model=16, n_layers=4, d_state=16, d_head=16, d_conv=4, include_time=True` |
| 5 | `family-83c4eb5a65db` | 0.5509 | 5 | 0.5789 | `lr=0.001, bs=32, d_model=64, n_layers=2, d_state=16, d_head=32, d_conv=4, include_time=False` |
