# midi-transformer
Transformer model to create midi songs

#### Data used
**Maestro v3.0** (all)
> Hawthorne, C., Stasyuk, A., Roberts, A., Simon, I., Huang, C.-Z. A., Dieleman, S., Elsen, E., Engel, J., & Eck, D. (2019). Enabling Factorized Piano Music Modeling and Generation with the MAESTRO Dataset. In International Conference on Learning Representations. Retrieved from https://openreview.net/forum?id=r1lYRjC9F7

**GiantMIDI-Piano** (all)
> Kong, Q., Li, B., Chen, J., & Wang, Y. (2020). GiantMIDI-Piano: A large-scale MIDI dataset for classical piano music. arXiv preprint arXiv:2010.07061. Retrieved from https://arxiv.org/pdf/2010.07061

**ADL Piano MIDI** (selection)
> Ferreira, L. N., Lelis, L. H. S., & Whitehead, J. (2020). Computer-Generated Music for Tabletop Role-Playing Games. In Proceedings of the 16th AAAI Conference on Artificial Intelligence and Interactive Digital Entertainment (AIIDE'20). Retrieved from https://github.com/lucasnfe/adl-piano-midi


#### Encoding
Each note is being encoded in only three parameters: Length, Velocity and Note

**Length**
Normalize 0. to 1. taking the maximum value in the sequence. The goal of this is to be agnostic of the speed the piece is being played. Also a scalling factor (10% by default) is being added to ensure the length can be higher in the output of the model.

**Velocity**
Normally encoded 0 to 127 in midi, simply normalized to 0 to 1 as an input to the model.

**Note**
Normally encoded 0 to 127 in midi, we add an extra input for wait time. It then
goes through an embedding layer to reduce from 128 one-hot encoding to fewer parameters (64 by default).

###### Example
The following midi sequence:
>0, 144, 60, 100 | Time 0, Note On,  C (60), velocity 100
>0, 144, 64, 100 | Time 0, Note On, E (64), velocity 100
>10, 128, 60, 0 | Time 10, Note Off, C (60), velocity 0
>10, 128, 64, 0 | Time 10, Note Off, C (64), velocity 0
>15, 144, 67, 100 | Time 15, Note On, G (67), velocity 100
>20, 128, 67, 0 | Time 20, Note Off, G (67), velocity 100

Would become
> 10, 100, 60   |   Play C (60) for Time 60 at Velocity 100
> 10, 100, 64   |  Play E (64) for Time 60 at Velocity 100
> 5, 0, 128   |   Wait (128) for Time 5
> 5, 100, 67    |   Play G (67) for Time 5 at Velocity 100

And then normalize the length (aka time) and the velocity from 0 to 1 and embbed the note to 64 parameters.

#### Model Architecture
Given access to "only" 4 P100 GPUs, the size of the model has been restricted to 2M parameters and consists of 14 relative attention layers with 12 heads and an embedding size of 64 for the note, length and velocity.

#### Help
If you have access to more processing power, would be interesting to train this model further or increase it's depth!

Enjoy!
