# Trainer 클래스 구현
from common.util import clip_grads
import numpy as np
import matplotlib.pyplot as plt
import time
plt.rc('font', family='AppleGothic')


class Trainer:
    """ 모델 학습, 학습 시간 측정 및 loss 시각화하는 클래스
    
    Args:
        model: 설계한 신경망 모델
        optimizer: 오차역전파로 구한 기울기를 기반으로 파라미터 갱신시킬 최적화 기법
        loss_list: 손실 값 저장 리스트
        eval_interval: 손실값 체크할 Epoch 간격
    
    """
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.loss_list = []
        self.eval_interval = None
        self.current_epochs = 0
        
    def fit(self, x, t, max_epochs=10, batch_size=10, max_grad=None, eval_interval=20):
        data_size = len(x)
        step_per_epoch = data_size // batch_size
        self.eval_interval = eval_interval
        model, optimizer = self.model, self.optimizer
        total_loss = 0
        loss_count = 0
        
        start_time = time.time()
        for epoch in range(max_epochs):
            # 전체 학습 데이터 shuffle
            idx = np.random.permutation(np.arange(data_size))
            x = x[idx]
            t = t[idx]
            
            # 1번의 Epoch 당 미니 배치 학습 시작
            for step in range(step_per_epoch):
                x_batch = x[step*batch_size: (step+1)*batch_size]
                t_batch = t[step*batch_size: (step+1)*batch_size]
                
                # 순전파, 손실 함수 계산, 역전파 수행
                loss = model.forward(x_batch, t_batch)
                model.backward()
                params, grads = remove_duplicate(model.params, model.grads) # 중복되는 가중치를 하나로 모음
                if max_grad is not None:
                    clip_grads(grads, max_grad)  # -> 기울기 노름이 이 값을 넘어서면 gradient clipping함(나중에 배울 예정)
                
                # 파라미터 갱신
                optimizer.update(params, grads)
                total_loss += loss
                loss_count += 1
                
                if (eval_interval is not None) and (step % eval_interval) == 0:
                    avg_loss = total_loss / loss_count
                    runtime = time.time() - start_time
                    print('| 에폭 %d |  반복 %d / %d | 시간 %d[s] | 손실 %.2f'
                          % (self.current_epochs + 1, step + 1, step_per_epoch, runtime, avg_loss))
                    self.loss_list.append(float(avg_loss))
                    total_loss, loss_count = 0, 0
            
            self.current_epochs += 1
                
    def plot(self, ylim=None):
        x = np.arange(len(self.loss_list))
        if ylim is not None:
            plt.ylim(*ylim)
        plt.plot(x, self.loss_list, label='train')
        plt.xlabel('반복 (x' + str(self.eval_interval) +')')
        plt.ylabel('손실')
        plt.show()

                

                
def remove_duplicate(params, grads):
    '''
    매개변수 배열 중 중복되는 가중치를 하나로 모아
    그 가중치에 대응하는 기울기를 더한다.
    '''
    params, grads = params[:], grads[:]  # copy list

    while True:
        find_flg = False
        L = len(params)

        for i in range(0, L - 1):
            for j in range(i + 1, L):
                # 가중치 공유 시
                if params[i] is params[j]:
                    grads[i] += grads[j]  # 경사를 더함
                    find_flg = True
                    params.pop(j)
                    grads.pop(j)
                # 가중치를 전치행렬로 공유하는 경우(weight tying)
                elif params[i].ndim == 2 and params[j].ndim == 2 and \
                     params[i].T.shape == params[j].shape and np.all(params[i].T == params[j]):
                    grads[i] += grads[j].T
                    find_flg = True
                    params.pop(j)
                    grads.pop(j)

                if find_flg: break
            if find_flg: break

        if not find_flg: break

    return params, grads     
