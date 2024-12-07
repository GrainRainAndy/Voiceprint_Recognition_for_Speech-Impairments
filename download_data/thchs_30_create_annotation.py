import argparse
import os
import functools
from utility import add_arguments, print_arguments


parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
parser.add_argument("--target_dir",
                    default="../dataset/audio/",
                    type=str,
                    help="存放音频文件的目录 (默认: %(default)s)")
parser.add_argument("--annotation_text",
                    default="../dataset/annotation/",
                    type=str,
                    help="存放音频标注文件的目录 (默认: %(default)s)")
parser.add_argument("--subset",
                    default="dev",
                    type=str,
                    help="使用的子数据集 (默认: %(default)s)")
args = parser.parse_args()


def create_annotation_text(data_dir, annotation_path, subset):
    if not os.path.exists(annotation_path):
        os.makedirs(annotation_path)
    print('Create THCHS-30-'+subset+' annotation text ...')
    f_a = open(os.path.join(annotation_path, 'thchs_30_'+subset+'.txt'), 'w', encoding='utf-8')
    data_path = subset
    for file in os.listdir(os.path.join(data_dir, data_path)):
        if '.trn' in file:
            file_path = os.path.join(data_dir, data_path, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                target_path = f.readline().strip()
            target_file_path = os.path.join(data_dir, data_path, target_path)
            with open(target_file_path, 'r', encoding='utf-8') as target_file:
                chinese_text = target_file.readline().strip()
                chinese_text = ''.join(chinese_text.split())
            f_a.write(file_path[3:-4]+ '\t' + chinese_text + '\n')
    f_a.close()
    print("Done.")


def prepare_dataset(target_dir, annotation_path, subset):
    data_dir = os.path.join(target_dir, 'data_thchs30')
    create_annotation_text(data_dir, annotation_path, subset)


def main():
    print_arguments(args)
    if args.target_dir.startswith('~'):
        args.target_dir = os.path.expanduser(args.target_dir)

    prepare_dataset(target_dir=args.target_dir,
                    annotation_path=args.annotation_text,
                    subset=args.subset)


if __name__ == '__main__':
    main()
