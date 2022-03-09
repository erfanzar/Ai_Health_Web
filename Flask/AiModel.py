import torch
import torch.nn as nn
import numpy as np  ##used for training
import matplotlib  ##used for training
import matplotlib.pyplot as plt  ##used for training
import os  ##used for training
import sys  ##used for training
import time  ##used for training
import sklearn as sk  ##used for training
import torch.optim as optim
import torch.nn.functional as F


MODEL_DICT_PATH = (
    "Flask/ModelCheckPoint/AIHealthT2ModelDict.pth"
)
OPTIMIZER_DICT_PATH = (
    "Flask/ModelCheckPoint/AIHealthT2OptimizerDict.pth"
)

MODEL_DICT_PATH_T4 = (
    "Flask/ModelCheckPoint/AiHealthT4ModelDict.pth"
)


OPTIMIZER_DICT_PATH_T4 = (
    "Flask/ModelCheckPoint/AiHealthT4OptimizerDict.pth"
)



MODEL_DICT_PATH_GRUANN = (
    "Flask/ModelCheckPoint/T9Model.pth"
)



OPTIMIZER_DICT_PATH_GRUANN = (
    "Flask/ModelCheckPoint/T9Optim.pth"
)


MODEL_DICT_PATH_GRUA = (
    "Flask/ModelCheckPoint/SavesAi/LST-lastReworkModel.pth"
)


OPTIMIZER_DICT_PATH_GRUA = (
    "Flask/ModelCheckPoint/SavesAi/LST-lastReworkModel.pth"
)



"""
    this model is trained only by defult parms on Google.colab research

"""


device = ('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)



class GRUA(nn.Module):
  def __init__(self, inputs , numberofhiddenlayers , numberoflayers , output , useReLU = True):
      super(GRUA , self).__init__()
      
      self.N = inputs

      self.M = numberofhiddenlayers

      self.L = numberoflayers

      self.K = output

      self.ReLU = useReLU

      self.Rnn = nn.GRU(self.N , self.M , self.L , batch_first=True).to(device)

      self.fc0 = nn.Linear(self.M , self.K*4).to(device)

      self.fc1 = nn.Linear(self.K*4 , self.K*3).to(device)

      self.fc2 = nn.Linear(self.K*3 , self.K*5).to(device)

      self.fc3 = nn.Linear(self.K*5 , self.K*3).to(device)

      self.fc4 = nn.Linear(self.K*3 , self.K*3).to(device)

      self.fc5 = nn.Linear(self.K*3 , self.K*2).to(device)

      self.fc6 = nn.Linear(self.K*2 , self.K).to(device)

  def forward(self , X):
    
    # try :

    X = X.view(-1,1,self.N)

    X = F.normalize(X)

    h0 = torch.randn((self.L , X.size(0) , self.M)).to(device)
    c0 = torch.randn((self.L , X.size(0) , self.M)).to(device)
    pred , _ = self.Rnn(X,h0)

    pred = self.fc0(F.relu(pred[:,-1,:]) if self.ReLU == True else pred[:,-1,:])

    pred = self.fc1(F.relu(pred) if self.ReLU == True else pred )

    pred = self.fc2(F.relu(pred) if self.ReLU == True else pred )

    pred = self.fc3(F.relu(pred) if self.ReLU == True else pred )

    pred = self.fc4(F.relu(pred) if self.ReLU == True else pred )

    pred = self.fc5(F.relu(pred) if self.ReLU == True else pred )

    pred = self.fc6(F.relu(pred) if self.ReLU == True else pred )

    return pred


    # except:

    #   raise RuntimeError('invalid input shape')




ModelGRUA = GRUA(7,80,10,4)

ModelGRUAOptimizer = optim.Adam(ModelGRUA.parameters() , lr = 0.0001)


ModelGRUA.load_state_dict(torch.load(MODEL_DICT_PATH_GRUA ,map_location=torch.device(device)))

# ModelGRUAOptimizer.load_state_dict(torch.load(OPTIMIZER_DICT_PATH_GRUA, map_location=torch.device(device)))



class ModelNet(nn.Module):
  def __init__(self, inputs , numberofhiddenlayers , numberoflayers , output , useReLU = True):
      super(ModelNet , self).__init__()

      self.N = inputs

      self.M = numberofhiddenlayers

      self.L = numberoflayers

      self.K = output

      self.ReLU = useReLU

      self.Rnn = nn.GRU(self.N , self.M , self.L)

      self.fc0 = nn.Linear(self.M , self.M*2)

      self.fc1 = nn.Linear(self.M*2 , self.M*3)

      self.fc2 = nn.Linear(self.M*3 , self.M*2)

      self.fc3 = nn.Linear(self.M*2 , self.M)

      self.fc4 = nn.Linear(self.M , self.K)

  def forward(self , X):

    try :

      X = X.view(-1,1,self.N)

    except:

      raise RuntimeError('invalid input shape')

    h0 = torch.randn((self.L , X.size(0) , self.M))

    pred , _ = self.Rnn(X,h0)

    pred = self.fc0(F.relu(pred[:,-1,:]) if self.ReLU == True else pred[:,-1,:])

    pred = self.fc1(F.relu(pred) if self.ReLU == True else pred )

    pred = self.fc2(F.relu(pred) if self.ReLU == True else pred )

    pred = self.fc3(F.relu(pred) if self.ReLU == True else pred )

    pred = self.fc4(F.relu(pred) if self.ReLU == True else pred )

    return pred




loss_functionModelByte = nn.L1Loss()






class GRUANN(nn.Module):
  def __init__(self, inputs , numberofhiddenlayers , numberoflayers , output , useReLU = True):
      super(GRUANN , self).__init__()

      self.N = inputs

      self.M = numberofhiddenlayers

      self.L = numberoflayers

      self.K = output

      self.ReLU = useReLU

      self.Rnn = nn.GRU(self.N , self.M , self.L , batch_first=True).to(device)

      self.fc0 = nn.Linear(self.M , self.K*4).to(device)

      self.fc1 = nn.Linear(self.K*4 , self.K*3).to(device)

      self.fc2 = nn.Linear(self.K*3 , self.K*5).to(device)

      self.fc3 = nn.Linear(self.K*5 , self.K*3).to(device)

      self.fc4 = nn.Linear(self.K*3 , self.K*3).to(device)

      self.fc5 = nn.Linear(self.K*3 , self.K*2).to(device)

      self.fc6 = nn.Linear(self.K*2 , self.K).to(device)

  def forward(self , X):

    try :

        X = X.view(-1,1,self.N)

        X = F.normalize(X)

        h0 = torch.randn((self.L , X.size(0) , self.M)).to(device)
        c0 = torch.randn((self.L , X.size(0) , self.M)).to(device)
        pred , _ = self.Rnn(X,h0)

        pred = self.fc0(F.relu(pred[:,-1,:]) if self.ReLU == True else pred[:,-1,:])

        pred = self.fc1(F.relu(pred) if self.ReLU == True else pred )

        pred = self.fc2(F.relu(pred) if self.ReLU == True else pred )

        pred = self.fc3(F.relu(pred) if self.ReLU == True else pred )

        pred = self.fc4(F.relu(pred) if self.ReLU == True else pred )

        pred = self.fc5(F.relu(pred) if self.ReLU == True else pred )

        pred = self.fc6(F.relu(pred) if self.ReLU == True else pred )

        return pred


    except:

      raise RuntimeError('invalid input shape')




GruAnn = GRUANN(7,20,2,4)

GruAnn.to(device)

GRUANNloss_function = nn.MSELoss().to(device)

GRUANNoptimizer = optim.Adam(GruAnn.parameters() , lr=0.0001)


modelStateDict = torch.load(MODEL_DICT_PATH_GRUANN ,map_location=torch.device('cpu'))

optimizerStateDict = torch.load(OPTIMIZER_DICT_PATH_GRUANN , map_location=torch.device('cpu'))

GruAnn.load_state_dict(modelStateDict)

GRUANNoptimizer.load_state_dict(optimizerStateDict)

def ppt(f):
    print(f)

class NET(nn.Module):
      def __init__(self , outputDimetions ,ReLULDrop=0.01):


          '''
            Dosnt have enought accuracy Acctually 70%
          '''

          super(NET , self).__init__()

          self.outd = outputDimetions


          self.fc001 = nn.Linear(1,2)

          self.fc002 = nn.Linear(1,2)

          self.fc003 = nn.Linear(1,2)

          self.fc004 = nn.Linear(1,2)

          self.fc005 = nn.Linear(1,2)

          self.fc006 = nn.Linear(1,2)

          self.fc007 = nn.Linear(1,2)

          self.fc011 = nn.Linear(1,2)

          self.fc022 = nn.Linear(1,2)

          self.fc033 = nn.Linear(1,2)

          self.fc044 = nn.Linear(1,2)

          self.fc055 = nn.Linear(1,2)

          self.fc066 = nn.Linear(1,2)

          self.fc077 = nn.Linear(1,2)

          self.fc111 = nn.Linear(2,1)

          self.fc222 = nn.Linear(2,1)

          self.fc333 = nn.Linear(2,1)

          self.fc444 = nn.Linear(2,1)

          self.fc555 = nn.Linear(2,1)

          self.fc666 = nn.Linear(2,1)

          self.fc777 = nn.Linear(2,1)

          self.fcl   = nn.Linear(7,1)

          self.lstm = nn.LSTM(7,30,2 , batch_first=True)

          self.fcp  = nn.Linear(30,1)


      def Forward(self ,g ,a , fo , sp,bo ,me ,sm):


        g = F.normalize(g.view(-1,1))

        a = F.normalize(a.view(-1,1))

        fo = F.normalize(fo.view(-1,1))

        sp = F.normalize(sp.view(-1,1))

        bo = F.normalize(bo.view(-1,1))

        me = F.normalize(me.view(-1,1))

        sm = F.normalize(sm.view(-1,1))

        gpred = self.fc001(F.relu(g))
        gpredout = self.fc011(F.relu(g))
        gtensor = torch.tensor((F.relu(gpred) *F.relu(gpredout)))
        gprediction = self.fc111(F.relu(gtensor))


        apred = self.fc002(F.relu(a))
        apredout = self.fc022(F.relu(a))
        atensor = torch.tensor((F.relu(apred) *F.relu(apredout)))
        aprediction = self.fc222(F.relu(atensor))


        fopred = self.fc003(F.relu(fo))
        fopredout = self.fc033(F.relu(fo))
        fotensor = torch.tensor((F.relu(fopred) *F.relu(fopredout)))
        foprediction = self.fc333(F.relu(fotensor))


        sppred = self.fc004(F.relu(sp))
        sppredout = self.fc044(F.relu(sp))
        sptensor = torch.tensor((F.relu(sppred) *F.relu(sppredout)))
        spprediction = self.fc444(F.relu(sptensor))


        bopred = self.fc005(F.relu(bo))
        bopredout = self.fc055(F.relu(bo))
        botensor = torch.tensor((F.relu(bopred) * F.relu(bopredout)))
        boprediction = self.fc555(F.relu(botensor))


        mepred = self.fc006(F.relu(me))
        mepredout = self.fc066(F.relu(me))
        metensor = torch.tensor((F.relu(mepred) * F.relu(mepredout)))
        meprediction = self.fc666(F.relu(metensor))


        smpred = self.fc007(F.relu(sm))
        smpredout = self.fc077(F.relu(sm))
        smtensor = torch.tensor((F.relu(smpred) * F.relu(smpredout)))
        smprediction = self.fc777(F.relu(smtensor))

        XArrow = torch.tensor((gprediction , aprediction ,foprediction , spprediction , boprediction ,meprediction,smprediction))
        GArrow = torch.tensor((g ,a , fo , sp,bo ,me ,sm))
        # h0 = torch.zeros((3,7,30))
        # c0 = torch.zeros((3,7,30))

        oop = F.normalize(GArrow.view(-1,1,7))

        h0 = torch.zeros((2,oop.size(0),30))
        c0 = torch.zeros((2,oop.size(0),30))

        # print(oop.size(0))

        out , _ = self.lstm(oop,(h0,c0))

        out = self.fcp(F.relu(out[:,-1,:]))


        return out







class Net(nn.Module):
    def __init__(self, inputd, output, hiddenRnn, layerRnn):

        super(Net, self).__init__()
        """
            taking input , output , hiddenRnn , layerRnn as args and requireds
            input is going to be input for the long short term memory or LSTM Layer Of the neuron
            output is going to be the out put of the last layer to calculate the Result of the Other Neurons
            hiddenRnn is the all the hidden layers in lstm
            layerRnn is the all the layers in lstm

            Defualt Params :
                inputd    (7)
                output    (1)
                hiddenRnn (20)
                layerRnn  (2)


            CopyRight MIT BY github:erfnazar
        """
        self.input = inputd

        self.M = hiddenRnn
        
        """sumary_line
        
        
        This model is based on lstm i tryed this one for somthing around 3 times on diffrend train prameters
        and got same result as always u can use the gru ann create line 113 name class with GRUA
        
        Return: return_descriptions
        """
        

        self.L = layerRnn

        self.output = output

        self.rnn = nn.LSTM(self.input, self.M, self.L, batch_first=True)

        self.fc0 = nn.Sequential(
            nn.Linear(self.M, 20), nn.ReLU(), nn.Linear(20, self.M), nn.ReLU()
        )

        self.fc1 = nn.Sequential(
            nn.Linear(self.M, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, self.output),
        )

    def Forward(self, X):

        self.X = F.normalize(X)

        h0 = torch.zeros((self.L, self.X.size(0), self.M))

        c0 = torch.zeros((self.L, self.X.size(0), self.M))

        self.X, _ = self.rnn(self.X, (h0, c0))

        # x = self.fc0(self.X[:,-1,:])

        # pred = self.fc1(x)

        pred = self.fc1(self.X[:, -1, :])

        return pred


Model = Net(7, 1, 20, 2)

loss_function = nn.L1Loss()

optimizer = optim.Adam(Model.parameters(), lr=0.001)

Model.load_state_dict(torch.load(MODEL_DICT_PATH))
optimizer.load_state_dict(torch.load(OPTIMIZER_DICT_PATH))
ModelByte = ModelNet(7,60,4,1)
OptimizerByte = optim.Adam(ModelByte.parameters() , lr=0.001)

ModelByte.load_state_dict(torch.load(MODEL_DICT_PATH_T4))
OptimizerByte.load_state_dict(torch.load(OPTIMIZER_DICT_PATH_T4))

def AiByte(X):
    X = np.array(X)
    X = [X]
    Xe = torch.from_numpy(np.float32(X))
    if (Xe.shape[2]) == 7:
        with torch.no_grad():
            pred = ModelByte.forward(Xe)
            return pred
    else:
        raise RuntimeError(
            f"Wrong input the input shape must be (inf , 1, 7) Yours were {Xe.shape()}"
        )

def Reverse(Numba , min , max):
    l = min
    m = max
    out = None
    for i in range(max):
        l +=1
        m -=1
        if Numba == i:
            out = m
    return out



def AiGruAnn(X):
    X = np.array(X)
    X = [X]
    Xe = torch.from_numpy(np.float32(X))
    if (Xe.shape[2]) == 7:
        with torch.no_grad():
            pred = GruAnn.forward(Xe)
            return pred
    else:
        raise RuntimeError(
            f"Wrong input the input shape must be (inf , 1, 7) Yours were {Xe.shape()}"
        )


def Ai(X):
    X = np.array(X)
    X = [X]
    Xe = torch.from_numpy(np.float32(X))
    if (Xe.shape[2]) == 7:
        with torch.no_grad():
            pred = Model.Forward(Xe)
            return pred
    else:
        raise RuntimeError(
            ('Wrong input the input shape must be (inf , 1, 7) Yours were' + Xe.shape())
        )


def AiGRUA(X):
    X = np.array(X)
    X = [X]
    Xe = torch.from_numpy(np.float32(X))
    if (Xe.shape[2]) == 7:
        with torch.no_grad():
            pred = ModelGRUA.forward(Xe)
            return pred
    else:
        raise RuntimeError(
            ('Wrong input the input shape must be (inf , 1, 7) Yours were %v ' , Xe.shape())
        )




def test():
    dumy_input = torch.rand((1, 1, 7))
    with torch.no_grad():
        pred = Model.Forward(dumy_input)
        print(f"X : {dumy_input} , Pred : {pred}")
