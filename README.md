---
title: 歷時語料庫
subtitle: diachronic corpus analysis
author: 
date: 2021-04-23
---



\chapter{歷時語料庫}

## 背景

早在1982年，語言學家Sinclair如此描繪了他對於未來語料庫模樣的想像，大量的文字藉由語料庫的建置被保存了下來，這些文字緩慢地、卻不停歇地演變，在語言使用之中留下演化的軌跡，等待我們發掘其中大大小小的變動。在參照語料庫（reference corpora）愈臻完善的同時，我們對歷時/史語料庫（diachronic/historical corpora）可以為我們帶來什麼，也感到好奇。

語言，將人們的所思所想傳遞、紀錄下來，並在說話者使用語言時，不斷被重塑與流傳。從共時（synchronic）的角度來看，語意存在各種變異（variation），而在歷時（diachronic）的脈絡下，經過時間累積而形成各種變遷。「歷時」所涵蓋的時間範圍可能是近代，也可能是距離我們非常遙遠的歷史時期。近年來文字在網路上大量流傳，加上社會快速變遷，語意表達亦不斷變化。與此同時，歷史文本的電子化數量的增長，使我們得以從中分析、挖掘詞彙所蘊含的詞意。

在歷時的主題下，我們可以將變項之間的關聯理解成時間本身就是一個變項，而其他的變項沿著時序縱線分佈。換個角度想，時間這個變項的資料點順序便成了關鍵，可以影響我們對於資料的看法，例如分群分析、詞向量相似度比較等等。

## 歷時語料庫

### 舉隅
中研院古漢語語料庫分別有(1) 上古漢語（Old Chinese，先秦到西漢）、(2) 中古漢語（Middle Chinese，東漢六朝）、以及(3) 近代漢語（Early Mandarin Chinese，唐朝至清朝）三個子語料庫，部分語料來源為漢籍全文資料庫（Scripta Sinica），主要提供漢語語法研究，並依此語料庫建立了詞彙庫。此外，中國語料庫在線也有现代汉语语料库及古代汉语语料库的對照。有關不同時間的中文使用，中國哲學書電子計畫（Chinese Text Project，簡稱[CTEXT](https://ctext.org)）所搜集的古文文本涵蓋朝代完整，文本的標點符號已有統一，且為同一文本嘗試了多種OCR辨識技術改善古文電子化的品質，並將已校對完成的文本放入「原典」，其餘則收錄於名為「維基」的資料區之下。

除此之外，COHA（Corpus of Historical American English）以及Google Ngram Book都是經典的歷時語料庫，COHA採用了COCA的語料庫建置方式，因此兩個斷代（分別為1810至2010年、1990至2010年）的語料庫便成了觀察英文語言演化極佳的來源，而Google Ngram據稱收集了世界上百分之六的書，在網頁介面輸入一或多個字詞，就可以查看、比較這些字詞在不同時間的使用頻率變化，但因著作權考量，僅開放查詢字詞前後（window size）五個字的語料；歐盟CLARIN（Common Language Resources and Technology Infrastructure）也有許多[歷史語料庫的介紹](https://www.clarin.eu/resource-families/historical-corpora)，其中包含了[Sheffield Corpus of Chinese](https://www.dhi.ac.uk/scc/)。

### 前處理

在歷時語意的主題下，現代語言的語料和古文的時間資訊細度（granularity）大不相同，網路上可以針對每十年、每月、甚至每天每小時等極短的時間間隔，收集到豐富連續的語料，即時追蹤趨勢。雖然古文文本的時間訊息不一定那麼完整，卻可以帶我們回到千年歷史。

手上握有了古漢語文本後，腦中浮現了另一個問題，古漢語文字使用與現代漢語的特色迥異，斷句、斷詞、詞性標記頓時成了關鍵又棘手的煩惱，甚至有研究打破了漢語斷詞必要性的想法（Meng et al., 2019）。即使如此，我們也有寶貴的資源可以參考，例如：目前有基於特定語料訓練而成的依存句法剖析器（dependency parser），如：[`UD-Kanbun`](https://pypi.org/project/udkanbun/)及`StandfordNLP`的[`Stanza`](https://stanfordnlp.github.io/stanza/)。

語料若是以Unicode編碼，在電腦上擁有其專屬的對應位置，或稱Unicode碼（code point），以及更大的Unicode區塊（block）。除了現代漢語常見的編碼範圍U+4E00至U+9FFF，在CJK Extension A 至 F、CJK Compatibility Ideographs亦有古漢語用字的蹤跡出現，如圖 @fig:ch15_chars。如果以正規表示式預設的四碼`[\u4e00-\u9fff]`來尋找漢字的話，可能會將「𧙀」（U+27640）這一漢字切成表情符號「❤」（U+2764）與數字「4」兩部分。解決方法之一是改用正規表示式的`\p`找到所屬的編碼區，抓取出座落於中日韓統合表意文字（CJK Unified Ideographs）的字。[^1] 如果語料是經過標點符號處理的，亦可透過CJK Symbols and Punctuations篩選出來。

![](../figures/ch15_chars.png)

[^1]: 有時候同一個Unicode編碼的字（character），在不同語言輸入法顯示的字符（glyph）不同，亦無法單憑Unicode區塊斷定該字為漢字，而是藉此方法找到書寫文字（script）相似的字。

最後，根據我們想要回答的研究問題，進一步過濾語料。例如，為了更有效率地訓練詞向量模型，我們是否選擇特定詞頻的字詞作為訓練詞向量的語料？高頻詞可能會被認為是不具語境的（not content-bearing），在不同文本的使用特色差異較小。
## 歷時詞向量

### 從歷時詞向量看語意變遷

歷時詞向量語意變遷許多著名的例子與新的科技發明連動，蘋果Apple、亞馬遜Amazon已與科技品牌、電商之意密不可分；詞彙也會隨著時事、歷史事件的發展而有語意更迭，過去的研究選定war、peace、stable、iraq、marijuana等字詞，觀察其鄰近詞在時間軸上的興衰。這些例子讓我們從跨越多個時空的語料中，探索語料如何反映當下的時空背景，更進一步讓我們發掘更多語意變遷可能的例子。

在歷時語料中，有些詞彙並無明顯的詞頻變化，是很穩定（少）使用的字詞，但其鄰近詞卻有所不同，所傳達的意義也不大一樣。Word2Vec 詞向量將一個個以文字符號代表的詞，轉換成一串串範圍0至1的連續數列，得以讓我們量化計算文字所承載的意思與表達。我們可以為不同時間範圍的語料訓練不同的詞向量，並對齊至同一空間，對齊方法有訓練後對齊[^2]，也可以循序疊加。

[^2]: (https://github.com/williamleif/histwords) 

不過，以共現（co-occurrence）為概念出發的語意表徵，除了在字詞的層次上發覺意義分佈的異同，其多義層次在語境詞向量（contextualized word embeddings）推出後，更進一步獲得發揮，我們可以從無限可能的語境中觀察哪個詞意改變了、哪個詞意不變。詞向量與語境詞向量讓我們看見不同層次的語意變遷現象，也讓我們思索語意變遷涵蓋了哪些面向。

將時間軸推回到更古早以前，詞向量的語意表徵就變得更具有時間性（temporal）且動態（dynamic）了，不過目前古文的預訓練模型不多，語境詞向量更是稀少，想想 BERT 是基於繁簡中文維基百科資料作為語料的，即使我們今天有了文言文的維基百科內容，所訓練而成的詞向量是代表著什麼樣的語言使用呢？因此，在歷時詞向量的語意表徵上，還是有針對不同時間範圍訓練而成的詞向量，不論是沒有加入語境訊息的、還是有豐富語境的詞向量，都提供我們探索過去的一個機會，或許這讓語意變遷這個研究主題更貼近資料為本的研究方法，古籍資料的電子化是彌足珍貴的語料。

此外，目前詞向量訓練評估的方式可能無法直接適用於歷時詞向量上，像是相似度任務、類比任務等，都高度仰賴了地名等專有名詞。在語意變遷標記資料方面，可從共時資料延伸產出歷時資料，也可從語言學文獻中的經典例子為佐證，如：actually、will 等語法化的詞彙，或是利用字典中obselete 作為標籤的方法。

以原始的向量比較歷時語意，稱作「一階向量（first-order embeddings）」，相對於二階向量（second-order embedding），是將某一字詞與該字詞鄰近詞的相似度數值，在兩個時間點下，各串連成一個數列，來代表這個字詞的歷時語意。依較廣泛或較為聚焦的視角，分別有取所有詞彙的全域法（global measure），或是取部分詞彙的部分法（local measure）組成二階向量。僅取 10 至 50 個左右的鄰近詞抓出細微又劇烈的語意變化，因為以整個語言來看，整體的語意關係是相對穩定的，而部分法可幫助我們抓取出鄰近詞變化較明顯的字詞，這些字詞有時候反映了真實發生的事件或具有文化意涵的關鍵字。

### 以 PTT 語料庫為例

接著，我們以PTT的語料為例，對比2005年至2020年「台灣」一詞在八卦版與女版的語意變遷現象。我們以每五年為間隔，訓練單一年份的Word2vec詞向量、對齊向量空間，並以t-SNE降維至二維平面上，以視覺化呈現該詞的鄰近詞變化。

```python
embed_2005.wv.most_similar('台灣', topn=5)
# Output: [(‘中國’, 0.8369995355606079), (‘美國’, 0.7618862390518188), (‘日本’, 0.7539179921150208), (‘發展’, 0.7530442476272583), (‘迪士尼’, 0.7488158345222473)]
```

```python
embed_2005['台灣']
# Output: array([ 8.07284042e-02, -1.51521474e-01, 2.04981357e-01, …
```

![](../figures/ch15_taiwan.png)

從圖中，「台灣」一詞的鄰近詞從2005年的「中國」、「企業」、「國際」、「發展」，到2015年出現了新詞「鬼島」、以及2020年的「普世」、「核心」等詞彙。僅僅二十年載的時間，詞彙的向量語意已不停地變動，再加上詞頻、tf-idf、keyness等，讓我們能夠勾勒出詞彙的語意歷史。

### 語意變遷的發現

近年來的歷史詞彙語意研究，從詞意的改變、新舊字詞的興衰，探索其背後的運作機制與認知層面，已開始摸索出語意變遷的規律性（regularities）。

1. **Law of Prototypicality**: 語意變遷的程度與字詞的原型（prototype）呈現負相關，越非典型的詞意，語意變遷的程度越高。利用 K-means 分群分析找出字詞的分群，而分群的中心點所代表的是詞彙原型，也可以說是沒有成詞（non-lexicalized）的詞彙原型，因為這個中心點是分群分析的計算結果，可以想像該點是抽象的、不存在的一個字詞，而已成詞的原型是最靠近此中心點的那一個點。將分群數 k 值降低，劃分出更大的分群聚類，以觀察詞彙之間的語意界線。

2. **Law of Conformity** 及 **Law of Innovation**: 前者的意思是語意變遷的程度與使用頻率呈現負相關；後者則是對於詞頻相同的字詞，詞義數越多，語意變遷的程度越高。關於詞義數的計算，可從詞向量的詞彙共現網路來計算詞彙的多義程度，接著再以迴歸分析找出語意變遷程度與詞頻、多義性的關聯。

3. **Law of Parallel Change** 及 **Law of Differentiation**: 兩者為對立的假設，從近義詞（near-synonym）的角度切入，發現 law of parallel change 比 law of differentiation 明顯，也就是說近義詞彼此的語意演變朝向相近的語意發展。如果兩個時間點共有的鄰近詞越多，詞意變得越來越相近。

以變異程度為基礎的近鄰群聚分析法（Variability-based Neighbor Clustering, [VNC](http://corpora.lancs.ac.uk/stats/toolbox.php)），是 Gries and Hilpert 提出的階層式分群分析法，由下而上地將時間點相連的資料點合併，合併規則依距離計算方法而定，例如以平均距離最小的兩點作為新的一聚類，並在下一步的合併視為新的一點。利用 VNC 分群分析法，我們可以綜合各觀察變項，如詞頻、tf-idf、keyness、詞向量、各式語意變遷量測值等，劃分出漢語詞彙的發展時代區分。
