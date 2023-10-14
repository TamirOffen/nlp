# RNNs - Recurrent Neural Networks
credit: [Patrick Loeber: PyTorch RNN Tutorial - Name Classification Using A Recurrent Neural Net](https://youtu.be/WEV61GmmPrk?si=l0INF4xULpy-xRar)

    If training vanilla neural nets is optimization over functions, training recurrent nets is optimization over programs.

## RNN Architecture:
RNNs are a class of neural networks that allow previous outputs to be used as inputs while having hidden states.

![rnn architecture](/RNN/imgs/rnn_arch.png)

Notice how RNNs allow us to operate over sequences of vectors.

$$
h_{t} \leftarrow \underbrace{\phi}_{\text{non-linearity}}(\underbrace{V \times h_{t-1}}_{\text{prev hidden state}} + \underbrace{U \times x_t}_{\text{current input}}) \\
o_t \leftarrow W \times h_t
$$

$h_t \leftarrow \underbrace{\phi}_{\text{non-linearity}}(\underbrace{V \times h_{t-1}}_{\text{prev hidden state}} + \underbrace{U \times x_t}_{\text{current input}})$

$o_t \leftarrow W \times h_t$

RNN API:
```python
rnn = RNN()
y = rnn.step(x) # x is an input vector, y is the rnn's output vector
```

Vanilla RNN step implementation:
```python
class RNN:
    # W, U, V ...
    def step(self, x):
        # hidden state update:
        self.h = np.tanh(np.dot(self.V, self.h) + np.dot(self.U, x))
        # compute output vector
        y = np.dot(self.W, self.h)
        return y
        
```

RNN Types:

![rnn arch types](/RNN/imgs/rnn_types.png)
Each rectangle is a vector and arrows represent functions (e.g. matrix multiply). Input vectors are in red, output vectors are in blue and green vectors hold the RNN's state (more on this soon). From left to right: (1) Vanilla mode of processing without RNN, from fixed-sized input to fixed-sized output (e.g. image classification). (2) Sequence output (e.g. image captioning takes an image and outputs a sentence of words). (3) Sequence input (e.g. sentiment analysis where a given sentence is classified as expressing positive or negative sentiment). (4) Sequence input and sequence output (e.g. Machine Translation: an RNN reads a sentence in English and then outputs a sentence in French). (5) Synced sequence input and output (e.g. video classification where we wish to label each frame of the video). Notice that in every case are no pre-specified constraints on the lengths sequences because the recurrent transformation (green) is fixed and can be applied as many times as we like.

## RNN Pros and Cons:
| Advantage | Drawback |
| --- | --- |
| Can process input of any length | Computation is slow|
| Model size does not change w.r.t. size of input | Difficult to access information from a long time ago|
| Computation takes into account historical information | Cannot consider future input for the current state|
| Model weights are shared across time | |

---

## Project: Name Classification Using a RNN
In `/RNN/data` there are a collection of last names from different countries. We want to build a model that will input a last name and will output the origin country. For example, if we pass in the last name "smith", out model should output English.

### Data Preprocessing:
TODO, utils.py file

### RNN:
The architecture of the RNN is as follows:

![name class arch](/RNN/imgs/name_class_arch.png)



