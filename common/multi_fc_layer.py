# Multi FC Layer 만드는 클래스
import numpy as np
from common.layers import *
from common.gradients import numerical_gradient
from collections import OrderedDict


class MultiLayerNet:
    """Multi FC(Fully Connected) Layer 만들기
    
    Parameters
    ----------
    input_size: 1차원으로 flatten 시킨 입력 크기
    hidden_size_list: 은닉층의 노드 수를 담은 리스트 e.g) [50, 100, 150]
    output_size: 최종 출력 크기
    activation: 활성화 함수
    weight_init_std: 파라미터의 표준편차 지정
        - 'relu' 또는 'he'로 지정하면 He 초기값으로 설정
        - 'sigmoid' 또는 'xaiver'로 지정하면 Xaiver 초기값으로 설정
    weight_decay_lambda: L2 norm으로 파라미터 정규화 항의 강도 설정
    
    """
    def __init__(self, input_size, hidden_size_list, output_size,
                activation='relu', weight_init_std='relu', weight_decay_lambda=0):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size_list = hidden_size_list
        self.hidden_layer_num = len(hidden_size_list)
        self.weight_decay_lambda = weight_decay_lambda
        # 파라미터 담을 딕셔너리
        self.params = {}
        
        # 파라미터 초기화
        self.__init_weight(weight_init_std)
        
        #==============
        # 신경망 계층 생성
        #==============
        activation_layer = {'sigmoid': Sigmoid, 'relu': Relu}
        self.layers = OrderedDict()
        # 행렬 곱 & 활성함수 계층 생성
        for idx in range(1, self.hidden_layer_num+1):
            self.layers[f'Affine({idx})'] = Affine(self.params[f'W{idx}'], self.params[f'b{idx}'])
            self.layers[f'Activation({idx})'] = activation_layer[activation]()
        # 마지막 계층은 Softmax-with-Loss 계층 생성
        idx += 1
        self.layers[f'Affine({idx})'] = Affine(self.params[f'W{idx}'], self.params[f'b{idx}'])
        self.last_layer = SoftmaxWithLoss()
            
    
    def __init_weight(self, weight_init_std):
        """ 가중치 초기화
        
        Parameters
        ----------
        weight_init_std: 가중치 초기화 설정 값(생성자 함수에서 설정하 인수)
        
        """
        all_size_list = [self.input_size] + self.hidden_size_list + [self.output_size] # [784, 50, 100, 150, 10]
        for idx in range(1, len(all_size_list)):
            scale = weight_init_std
            if str(weight_init_std).lower() in ('relu', 'he'):
                # He normalization으로 설정
                scale = np.sqrt(2.0 / all_size_list[idx-1])
            elif str(weight_init_std).lower() in ('sigmoid'):
                scale = np.sqrt(1.0 / all_size_list[idx-1])
            elif str(weight_init_std).lower() in ('xaiver'):
                # Xaiver normalization으로 설정
                scale = np.sqrt(2.0 / (all_size_list[idx-1] + all_size_list[idx]))
            self.params[f'W{idx}'] = scale * np.random.randn(all_size_list[idx-1], all_size_list[idx])
            self.params[f'b{idx}'] = np.zeros(all_size_list[idx])

        
    def predict(self, x):
        """ 마지막 층(Softmax 레이어 전까지) 순전파 수행
        
        Parameters
        ----------
        x: 입력 데이터(또는 레이어들 사이의 중간 입력 데이터)
        
        """
        for layer in self.layers.values():
            x = layer.forward(x)
            
        return x
    
    
    def loss(self, x, t):
        """ 손실함수 값 계산
        
        Parameters
        ----------
        x: 입력 데이터
        t: 정답 레이블
        
        """
        y = self.predict(x) 
        
        # 정규화 항(weight decay)는 마지막 레이어의 출력값에만 더해주어 파라미터 감소시켜주는 것!
        weight_decay = 0
        for idx in range(1, self.hidden_layer_num+2):
            W = self.params[f'W{idx}']
            weight_decay += 0.5 * self.weight_decay_lambda * np.sum(W**2)  # L2 norm
            
        return self.last_layer.forward(y, t) + weight_decay
    
    
    def accuracy(self, x, t):
        """ 모델의 정확도 계산
        
        Parameters
        ----------
        x: 입력 데이터
        t: 정답 레이블
        
        """
        # 정확도 예측하기 위해서 일종의 '테스트'이므로 손실함수를 구할 필요가 없기 때문에 Softmax 전까지만 순전파 수행해도 됨
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim == 1:
            t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y == t) / float(y.shape[0])
        return accuracy
    
    
    def numerical_gradient(self, x, t):
        """ 수치미분 계산 for 오차역전파로 수행한 기울기 검증 위함
        
        Parameters
        ----------
        x: 입력 데이터
        t: 정답 레이블
        
        """
        loss_W = lambda w: self.loss(x, t)
        
        grads = {}
        for idx in range(1, self.hidden_layer_num+2):
            grads[f'W{idx}'] = numerical_gradient(loss_W, self.params[f'W{idx}'])
            grads[f'b{idx}'] = numerical_gradient(loss_W, self.params[f'b{idx}'])
        
        return grads
    
    
    def gradient(self, x, t):
        """ 오차역전파 수행
        
        Parameters
        ----------
        x: 입력 데이터
        t: 정답 레이블
        
        """
        # 순전파 수행하면서 손실함수 값 계산
        self.loss(x, t)
        
        # 역전파 수행
        dout = 1
        dout = self.last_layer.backward(dout)  # Softmax-with-Loss 역전파
        
        layers = list(self.layers.values())    # 나머지 앞의 계층들 역전파
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
            
        # 오차역전파로 구한 기울기 저장
        grads = {}
        for idx in range(1, self.hidden_layer_num+2):
            # 가중치의 기울기에는 정규화 항 적용한 후 저장
            grads[f'W{idx}'] = self.layers[f'Affine({idx})'].dW + self.weight_decay_lambda * self.layers[f'Affine({idx})'].W
            grads[f'b{idx}'] = self.layers[f'Affine({idx})'].db
        
        return grads
