from flask import Flask, render_template, request
from AiModel import Ai , AiByte ,AiGruAnn , ppt , AiGRUA
import numpy as np
# from Functions import Reverse

app = Flask(__name__)

Address = "192.168.1.108"



@app.route("/", methods=["GET", "POST"])
def home():
    ListFood = ['Fat' , "Dairy" , "Protein" , 'Fruit and vegetables']
    ListBlood = ["I Don't Have", "I Have"]
    ListSmok = ["i Don't", "i Do"]
    ListDrug = ["NO i didn't" , "yes i did"]
    bg = [
    'static/style/WallPaper-BackGround/Nature.jpg',
    'static/style/WallPaper-BackGround/LL_AI_feat.jpg',
    'static/style/WallPaper-BackGround/ai.png'
    ]

    return render_template(
        "InformationTaker.html",
        v1=False,
        ListFood=ListFood,
        ListBlood=ListBlood,
        ListDrug=ListDrug,
        bg= bg,

        ListSmok=ListSmok,
        Address=Address,
        NFood=0,
        NSport=0,
        NDoctor=0
    )


@app.route("/taker", methods=["POST"])
def taker():

    ListFood = ['Fat' , "Dairy" , "Protein" , 'Fruit and vegetables']
    ListBlood = ["I Don't Have", "I Have", "I Don't Know"]
    ListMental = ["I'm Good As Always", "I'm Feeling Sad", "Depression"]
    ListDrug = ["NO i didn't" , "yes i did"]
    ListSmok = ["i Don't", "i Do"]

    bg = [
    'static/style/WallPaper-BackGround/Nature.jpg',
    'static/style/WallPaper-BackGround/LL_AI_feat.jpg',
    'static/style/WallPaper-BackGround/ai.png'
    ]

    data = []

    if request.method == "POST":

        data = []

        Gender = request.form.get("Gender")

        Age = request.form.get("Age")

        Food = request.form.get("Food")

        Sport = request.form.get("Sport")

        Blood = request.form.get("Blood")

        Mental = request.form.get("Mental")

        Smok = request.form.get("Smok")


        data.append([Gender, Age, Food, Sport, Blood, Mental, Smok])


        NDoctor , NSport , NFood , Pred = AI(data)


    return render_template(
        "InformationTaker.html",
        v1=False,
        ListDrug=ListDrug,
        ListFood=ListFood,
        ListBlood=ListBlood,
        ListMental=ListMental,
        ListSmok=ListSmok,
        Address=Address,
        bg= bg,
        Pred=Pred,
        NFood=NFood,
        NSport=NSport,
        NDoctor=NDoctor
    )

def AI(data):


    pred = AiGRUA(data)


    preds = np.int32(np.array(pred[0][0]))

    Mental = data[0][5]

    Sport = data[0][3]


    LifeScore = preds

    if Mental == 0:

        LifeScore = preds

    if Mental == 1:
        LifeScore = preds-5

    if Mental == 2:
        LifeScore = preds-np.exp(preds)
    NFood = pred[0][1]

    NSport= pred[0][2]

    NDoctor = pred[0][3]



    return NDoctor , NSport , NFood , preds

if __name__ == "__main__":
    app.run(debug=True, host=Address)
