# Installation cohorts

Install this checkout in a fresh Python 3.11 or later environment. It pins core
0.15.2 at `ec1161f5d74481be3540cb0dfa63d0e8f8b54e90` and ParticleGAN 0.11.1.

```sh
python -m pip install '.[response-fixture,test]'
```

Native AR responses use the explicit continuous penalty domain. Read [current native setup](NATIVE-CONTINUOUS.md).

Replay older FULL/RAW artifacts from the matching product revision and environment.
The [archived installation guide](https://github.com/HyperGAN/music3-particle-sliders/blob/f23889b4fafd6a63efaff2ceeb7fc4c312f9f5d6/INSTALL-COHORTS.md)
retains their exact source and dependency pins.
