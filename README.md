# Semi Supervised Learning : Improving speech recognition accuracy using sythesized speech output using GAN

In this project we use Deep learning to synthesis speech/audio using WaveGAN and SpecGAN ([paper](https://arxiv.org/abs/1802.04208)). The thus synthesized raw audio is used for improving the baseline system. 

## Getting Started

### Prerequisites

* Tensorflow >= 1.4
* Python 3.6

### Datasets

### Baseline Speech Recognition System

We are using the isolated word recognition with a vocab size of 10. [baseline](https://www.tensorflow.org/tutorials/audio_recognition)

### GAN based Speech Synthesis system



### References

* Donahue, Chris, Julian McAuley, and Miller Puckette. "Synthesizing Audio with Generative Adversarial Networks." arXiv preprint arXiv:1802.04208 (2018). [paper](https://arxiv.org/abs/1802.04208)
* Shen, Jonathan, et al. "Natural TTS synthesis by conditioning wavenet on mel spectrogram predictions." arXiv preprint arXiv:1712.05884 (2017). [paper](https://arxiv.org/pdf/1712.05884.pdf)
* Perez, Anthony, Chris Proctor, and Archa Jain. Style transfer for prosodic speech. Tech. Rep., Stanford University, 2017. [paper](http://web.stanford.edu/class/cs224s/reports/Anthony_Perez.pdf)
* Goodfellow, Ian, et al. "Generative adversarial nets." Advances in neural information processing systems. 2014. [paper](https://arxiv.org/pdf/1406.2661.pdf)
* Salimans, Tim, et al. "Improved techniques for training gans." Advances in Neural Information Processing Systems. 2016. [paper](https://arxiv.org/pdf/1606.03498.pdf)
* Grinstein, Eric, et al. "Audio style transfer." arXiv preprint arXiv:1710.11385 (2017). [paper](https://arxiv.org/abs/1710.11385)
* Pascual, Santiago, Antonio Bonafonte, and Joan Serra. "SEGAN: Speech enhancement generative adversarial network." arXiv preprint arXiv:1703.09452 (2017). [paper](https://arxiv.org/pdf/1703.09452.pdf)
* Yongcheng Jing, Yezhou Yang, Zunlei Feng, Jingwen Ye, Yizhou Yu, Mingli Song. "Neural Style Transfer: A Review" 	arXiv:1705.04058 (2017) [paper](https://arxiv.org/abs/1705.04058v6)



## Authors

* **Anoop Toffy** - *IIIT Bangalore* - [Personal Website](www.anooptoffy.com)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Dr. Gue Jun Jung (Phd),Speech Recognition Tech, SK Telecom
* Google Korea
* Tensorflow Korea
* SK Telecom
