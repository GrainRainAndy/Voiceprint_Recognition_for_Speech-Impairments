import argparse
import functools
import time
import wave

from masr.predict import MASRPredictor
from masr.utils.utils import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('model_dir',        str,    'models/ConformerModel_fbank/inference_model/', "导出的预测模型文件夹路径")
add_arg('wav_path',         str,    'dataset/test.wav',            "预测音频的路径")
add_arg('is_long_audio',    bool,   False,                         "是否为长语音")
add_arg('real_time_demo',   bool,   False,                         "是否使用实时语音识别演示")
add_arg('use_gpu',          bool,   True,                          "是否使用GPU预测")
add_arg('use_pun',          bool,   False,                         "是否给识别结果加标点符号")
add_arg('is_itn',           bool,   False,                         "是否对文本进行反标准化")
add_arg('decoder',          str,    'ctc_beam_search',             "解码器，支持ctc_greedy、ctc_beam_search")
add_arg('decoder_configs',  str,    'configs/chinese_decoder.yml', "解码器配置参数文件路径")
add_arg('pun_model_dir',    str,    'models/pun_models/',          "加标点符号的模型文件夹路径")
args = parser.parse_args()
print_arguments(args=args)

# 获取识别器
predictor = MASRPredictor(model_dir=args.model_dir,
                          use_gpu=args.use_gpu,
                          decoder=args.decoder,
                          decoder_configs=args.decoder_configs,
                          use_pun=args.use_pun,
                          pun_model_dir=args.pun_model_dir)


# 短语音识别
def predict_audio():
    start = time.time()
    result = predictor.predict(audio_data=args.wav_path, use_pun=args.use_pun, is_itn=args.is_itn)
    score, text = result['score'], result['text']
    print(f"消耗时间：{int(round((time.time() - start) * 1000))}ms, 识别结果: {text}, 得分: {int(score)}")


# 长语音识别
def predict_long_audio():
    start = time.time()
    result = predictor.predict_long(audio_data=args.wav_path, use_pun=args.use_pun, is_itn=args.is_itn)
    score, text = result['score'], result['text']
    print(f"长语音识别结果，消耗时间：{int(round((time.time() - start) * 1000))}, 得分: {score}, 识别结果: {text}")


# 实时识别模拟
def real_time_predict_demo():
    # 识别间隔时间
    interval_time = 0.5
    CHUNK = int(16000 * interval_time)
    # 读取数据
    wf = wave.open(args.wav_path, 'rb')
    channels = wf.getnchannels()
    samp_width = wf.getsampwidth()
    sample_rate = wf.getframerate()
    data = wf.readframes(CHUNK)
    # 播放
    while data != b'':
        start = time.time()
        d = wf.readframes(CHUNK)
        result = predictor.predict_stream(audio_data=data, use_pun=args.use_pun, is_itn=args.is_itn, is_end=d == b'',
                                          channels=channels, samp_width=samp_width, sample_rate=sample_rate)
        data = d
        if result is None: continue
        score, text = result['score'], result['text']
        print(f"【实时结果】：消耗时间：{int((time.time() - start) * 1000)}ms, 识别结果: {text}, 得分: {int(score)}")
    # 重置流式识别
    predictor.reset_stream()


if __name__ == "__main__":
    if args.real_time_demo:
        real_time_predict_demo()
    else:
        if args.is_long_audio:
            predict_long_audio()
        else:
            predict_audio()
