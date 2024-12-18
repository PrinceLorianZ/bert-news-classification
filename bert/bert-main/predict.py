from model import MyModel
from config import parsers
import torch
from transformers import BertTokenizer
import time


def load_model(device, model_path):
    myModel = MyModel().to(device)
    myModel.load_state_dict(torch.load(model_path))
    myModel.eval()
    return myModel


def process_text(text, bert_pred):
    tokenizer = BertTokenizer.from_pretrained(bert_pred)
    token_id = tokenizer.convert_tokens_to_ids(["[CLS]"] + tokenizer.tokenize(text))
    mask = [1] * len(token_id) + [0] * (args.max_len + 2 - len(token_id))
    token_ids = token_id + [0] * (args.max_len + 2 - len(token_id))
    token_ids = torch.tensor(token_ids).unsqueeze(0)
    mask = torch.tensor(mask).unsqueeze(0)
    x = torch.stack([token_ids, mask])
    return x


def text_class_name(pred):
    result = torch.argmax(pred, dim=1)
    result = result.cpu().numpy().tolist()

    # Read the classification file and map it to English categories
    classification = open(args.classification, "r", encoding="utf-8").read().split("\n")
    classification_dict = dict(zip(range(len(classification)), classification))

    # Translate categories to English if needed
    translation_dict = {
        "体育": "Sports",  # Example translation of Chinese category to English
        "娱乐": "Entertainment",
        "家居": "Home",
        "房产": "Real Estate",
        "教育": "Education",
        # Add more translations as necessary
    }

    predicted_category = classification_dict[result[0]]
    translated_category = translation_dict.get(predicted_category,
                                               predicted_category)  # Use translated category if available

    # print(f"Text: {text}\tPredicted Category: {translated_category}")
    print(f"Text:Predicted Category: {translated_category}")


if __name__ == "__main__":
    start = time.time()
    args = parsers()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model = load_model(device, args.save_model_best)

    texts = ["麦基砍28+18+5却充满寂寞 纪录之夜他的痛阿联最懂新浪体育讯上天对每个人都是公平的，贾维尔-麦基也不例外。", "李冰冰爆成龙谈情害羞 房祖名不赞成父亲转型新浪娱乐讯 北京时间9月17日消息，", "厂家撤场客户无货提 好美家收消费者仓库保管费荆楚网消息 (楚天都市报) (记者张乐克 通讯员陆红林 肖军)宋先生预缴6000多元在好美家徐东路店购买瓷砖，提货时才被告知厂家已经撤场。",
             "楼市局部金九：二线城市成交大涨一线城市楼市成交量逐步萎缩，渐渐面临有价无市的窘境。而二三线城市的楼市却在逐渐升温，大范围展开补涨。地产大鳄和资本大佬也嗅到这一重大历史机遇，频繁出手。", "美国名校华裔学生和父母畅谈家庭教育经新华网华盛顿7月20日电 美国北卡罗来纳州华人企业联合会近日邀请美国名校的数名华裔学生及他们的父母参加座谈会，畅谈家庭教育的成功经验。"]
    texts_en = ["McGee scores 28+18+5 but is filled with loneliness. On this record night, his pain is best understood by Yi Jianlian. Sina Sports reports: 'Heaven is fair to everyone, and JaVale McGee is no exception.","Li Bingbing reveals Jackie Chan feels shy talking about romance. Fang Zuming disagrees with his father's career change. Sina Entertainment reports: 'On September 17th, Beijing time.'",
                "Manufacturer pulls out and customers are left without goods. Haomeijia charges consumers warehouse storage fees. Jingchu Net reports: 'Mr. Song prepaid over 6,000 yuan to buy tiles at Haomeijia's Xudong Road store, but was only informed at pickup that the manufacturer had already withdrawn.'","Real estate market sees a partial boom in September: Transactions in second-tier cities soar, while first-tier cities' transactions gradually shrink, facing the dilemma of having prices but no market. Meanwhile, the real estate market in second and third-tier cities is heating up, with widespread catch-up in prices. Real estate tycoons and capital moguls are also sensing this significant historical opportunity and are frequently taking action.",
                "Chinese-American students from prestigious U.S. universities and their parents talk about family education experiences. Xinhuanet, Washington, July 20th: 'The Chinese-American Business Association of North Carolina recently invited several Chinese-American students from top U.S. universities and their parents to participate in a seminar to share their successful family education experiences.'"]
    print("Model prediction results：")
    i = 0
    for text in texts:
        x = process_text(text, args.bert_pred)
        with torch.no_grad():
            pred = model(x)
        print(texts_en[i])
        i = i+1
        text_class_name(pred)
    end = time.time()
    print(f"The time taken is：{end - start} s")
