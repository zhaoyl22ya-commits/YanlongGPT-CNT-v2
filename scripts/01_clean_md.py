import os
import re
import json
from pathlib import Path


# ==============================
# 路径设置
# ==============================

# 当前文件位于：
# 项目根目录/scripts/01_clean_md.py
# 因此 parent.parent 就是项目根目录
BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = BASE_DIR / "data" / "raw" / "CNT_md_original"

OUT_DIR = BASE_DIR / "data" / "processed" / "CNT_clean"

LOG_FILE = BASE_DIR / "data" / "processed" / "clean_log.json"


OUT_DIR.mkdir(parents=True, exist_ok=True)


# ==============================
# 清洗函数
# ==============================


def remove_copyright(text):
    """
    删除版权声明区域
    """

    patterns = [

        r"This article.*?rights reserved.*?\n",

        r"Copyright.*?\n",

        r"Downloaded by.*?\n",

        r"All rights reserved.*?\n",

        r"This is a PDF file.*?\n",

        r"Accepted Manuscript.*?\n",

    ]

    count = 0

    for p in patterns:

        matches = re.findall(
            p,
            text,
            flags=re.I | re.S
        )

        count += len(matches)

        text = re.sub(
            p,
            "",
            text,
            flags=re.I | re.S
        )

    return text, count



def remove_headers(text):
    """
    删除页眉页脚
    """

    count = 0


    patterns = [

        # Page 1 / Page 1 of 20
        r"^\s*Page\s+\d+(\s+of\s+\d+)?\s*$",

        # 单独页码
        r"^\s*\d{1,4}\s*$",

        # 常见期刊页眉
        r"^\s*(Nature|Science|Elsevier|Springer|Wiley|ACS|RSC).*?$"

    ]


    lines = text.split("\n")

    new_lines=[]


    for line in lines:

        remove=False

        for p in patterns:

            if re.match(
                p,
                line,
                flags=re.I
            ):

                remove=True
                count+=1
                break


        if not remove:

            new_lines.append(line)



    return "\n".join(new_lines), count




def clean_unicode(text):
    """
    清理PDF转换产生的控制字符
    """

    count=0

    result=[]


    for c in text:


        # 删除不可见控制字符

        if ord(c)<32:

            if c in ["\n","\t"]:

                result.append(c)

            else:

                count+=1

        else:

            result.append(c)



    text="".join(result)



    # 连续异常符号

    text=re.sub(
        r"[�]{2,}",
        "[NOISE]",
        text
    )


    return text,count




def clean_markdown(text):

    count=0


    # 删除图片路径
    text=re.sub(
        r"!\[.*?\]\(.*?\)",
        "",
        text
    )


    # 删除html标签

    text=re.sub(
        r"<[^>]+>",
        "",
        text
    )


    # 空格整理

    text=re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )


    return text,count




def clean_article(text):

    log={}


    text,c=remove_copyright(text)

    log["copyright_removed"]=c


    text,c=remove_headers(text)

    log["header_removed"]=c


    text,c=clean_unicode(text)

    log["unicode_removed"]=c


    text,c=clean_markdown(text)

    log["markdown_removed"]=c


    return text,log



# ==============================
# 主程序
# ==============================


def main():


    logs=[]


    files=list(
        RAW_DIR.glob("*.md")
    )


    print(
        f"发现 {len(files)} 个md文件"
    )



    for i,file in enumerate(files):


        print(
            f"[{i+1}/{len(files)}] {file.name}"
        )


        try:

            with open(
                file,
                "r",
                encoding="utf-8",
                errors="ignore"
            ) as f:

                text=f.read()



            clean_text,info=clean_article(text)



            output_file=OUT_DIR / file.name


            with open(
                output_file,
                "w",
                encoding="utf-8"
            ) as f:

                f.write(clean_text)



            info["file"]=file.name

            info["original_chars"]=len(text)

            info["clean_chars"]=len(clean_text)


            logs.append(info)



        except Exception as e:

            print(
                "ERROR:",
                file.name,
                e
            )



    with open(
        LOG_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            logs,
            f,
            ensure_ascii=False,
            indent=2
        )


    print("\n===== 清洗完成 =====")

    print(
        "输出:",
        OUT_DIR
    )

    print(
        "日志:",
        LOG_FILE
    )



if __name__=="__main__":

    main()