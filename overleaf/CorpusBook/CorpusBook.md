早在\citeyear{sinclair1982reflections}，語言學家\citeauthor{sinclair1982reflections}如此描繪了對於未來語料庫模樣的想像，文字的保存是大量的，其中是緩慢卻不斷變動的語料（"vast, slowly changing stores of text"），亦是對語言演化很詳細的紀錄（"detailed evidence of language evolution"）。

語言，將人們的所思所想傳遞、紀錄下來，並在說話者使用語言時，不斷被重塑與流傳 \parencite[61]{blank1999new}。從共時（synchronic）的角度來看，語意存在各種變異（variation），而在歷時（diachronic）的脈絡下，經過時間累積而則彰顯了各種的變遷。「歷時」所代表的時間範圍可能是近代，也可能是距離我們非常遙遠的歷史時期，近年來文字在網路上大量流傳，加上社會快速變遷，語意表達亦不斷變化。與此同時，歷史文本的電子化數量的增長，使我們得以從中分析、挖掘詞彙所蘊含的詞意。

語料庫作為語言使用的經驗素材，提供了我們從中觀察、歸納出可質化、量化的語言分析，從前述章節提到的keyness及詞向量等概念，我們可以將感興趣的字詞語意帶入時間的軸度，開展了更多與歷時語意相關的研究可能，從語料量化與計算的觀點切入詞彙語意變遷的語言現象。而歷時語料庫更因應科技進步，結合了計算語言學界近年來的語言向量表徵、神經語言統計模型等新方式探求語意在時間洪流下的變動與趨勢。

## 歷時語料庫
- [Corpus of Historical American English (COHA, 1810-2010)](https://www.english-corpora.org/coha/)
- [Corpus of Historical American English (COHA, 1810-2010)]( https://www.english-corpora.org/coha/)
- [A Representative Corpus of Historical English Registers (ARCHER, 1600-1999)](https://www.projects.alc.manchester.ac.uk/archer/)
- [Corpus of Late Modern English Texts (CLMET, 1710-1920)](https://perswww.kuleuven.be/~u0044428/)
- [Sheffield Corpus of Chinese](https://www.dhi.ac.uk/scc/)
- [Academia Sinica Tagged Corpus of Old Chinese (中央研究院上古漢語語料庫, from pre-Qing to pre-Han)](http://lingcorpus.iis.sinica.edu.tw/ancient/), [Academia Sinica Tagged Corpus of Middle Chinese (中央研究院中古漢語語料庫, from late-Han to the Six Dynasties)](http://lingcorpus.iis.sinica.edu.tw/middle/), and [Academia Sinica Tagged Corpus of Early Mandarin Chinese (中央研究院近代漢語語料庫, from Tang to Qing)](http://lingcorpus.iis.sinica.edu.tw/early/). The division into 3 corpora is based on the development of Chinese syntax to offer a synchronic sketch of Chinese and a basis for diachronic comparisons. In the 3 Academia Sinica tagged corpora, raw texts are available, with part of the texts imported from Scripta Sinica (漢籍全文資料庫計畫). It is also worth noting that the Google Books project for Chinese is not available until the year of 1950, and the latest date is 2008. It is believed that corpora creation is the foundation for a more thorough and accurate depiction for data collection during the establishment of lexical databases.

除此之外，Google Ngram Book大量文本加上詳細的年份資訊，也很適合歷時的主題，但因著作權考量，僅開放window size為5的語料使用；歐盟CLARIN（Common Language Resources and Technology Infrastructure）也有許多歷史語料庫的介紹，網址是：https://www.clarin.eu/resource-families/historical-corpora。

在分析歷時語意之前，需要尋找到含有時間資訊的語料文本，以及對語料進行前處理，成為我們可以分析的歷時語料庫。如果是現代的語料，時間資訊細度（granularity）很高，可以針對每十年、每年、甚至每個月等很短的時間間隔，收集到豐富連續的語料，且在語料處理上擁有很多資源能夠讓我們使用。
中古、近代漢語與現代漢語的特色迥異，雖然不如現代漢語語料處理方便，亦已有寶貴的資源可以參考，例如：(1) 語料若是Unicode的形式，除了現代漢語使用的U+4E00至U+9FFF區塊（block），亦有CJK Extension A至F、CJK Compatibility Ideographs以及CJK Compatibility Ideographs可以找到古漢語中常出現的稀有字。(2) 儘管古漢語沒有採行標準化的標點符號系統，我們仍可透過Unicode的CJK Symbols and Punctuations篩選出語料電子化所加上的標點符號。(3) 斷詞是古漢語語料處理的一大困難處，目前有基於特定語料訓練而成的依存句法剖析器（dependency parser），如：[`UD-Kanbun`(Yasuoka, 2019)](https://pypi.org/project/udkanbun/)及`StandfordNLP`的[`Stanza`(Qi et al., 2020)](https://stanfordnlp.github.io/stanza/)。
```r
# 以下是混合了CJK Extension區塊的一些漢字，可以用\\p{Blk=<block>}的方式抓取出符合的結果。
library(stringr)

str_extract_all("䕫滋𬺵哈㐴虁𠓲𫝅𫠧𪜅", "[\\p{Unified_Ideograph}\\p{Blk=CJK}\\p{Blk=CJKExtA}\\p{Blk=CJKExtB}\\p{Blk=CJKExtC}\\p{Blk=CJKExtD}\\p{Blk=CJKExtE}\\p{Blk=CJKExtF}\\p{Blk=CJKCompatIdeographs}\\p{Blk=CJKCompatIdeographsSup}]")

str_extract_all("䕫滋𬺵哈㐴虁𠓲𫝅𫠧𪜅", "[\\p{Blk=CJKExtA}\\p{Blk=CJKExtB}\\p{Blk=CJKExtC}\\p{Blk=CJKExtD}\\p{Blk=CJKExtE}\\p{Blk=CJKExtF}\\p{Blk=CJKCompatIdeographs}\\p{Blk=CJKCompatIdeographsSup}]")
```

```python
import unicodedata
```
## 歷時詞向量
在歷時語料中，有些詞彙並無明顯的詞頻變化，其多義行為亦造成研究者面對巨量資料時的困擾。結合語料統計模型與計算語意學的表徵模型，探究漢語的語意變遷。從數位化的原始語料中，以共現（co-occurrence）分佈的趨勢發覺意義分布的異同，以量化的方式量測語意變遷的程度，並以質化分析輔證已知的例子，並發掘更多可能的例子與規律。

一階向量（first-order embedding）由詞向量模型的原始數值組成，例如：以Word2Vec訓練而成的300維向量。\textcite{hamilton2016cultural}提出以二階向量（second-order embedding）計算語意變遷的程度，將某字詞與其鄰近詞（neighboring word）的相似度串連成數列，來代表這個字詞的語意表徵，更可依據是否取其所有鄰近詞，抑或是部分鄰近詞，細分成全域法（global measure）及部分法（local measure）[^1]，因為以整個語言來看，語意是相對穩定的，而部分法可幫助我們抓取出語意變化較明顯的鄰近詞區段。

[^1]: 從\textcite{hamilton2016cultural}的研究結果中，發現25至50個鄰近詞即可。

以「台灣」一詞為例，從PTT 2005至2020年每五年的語料，看看八卦版及女版的鄰近詞變化，可發現八卦版在2005的鄰近詞有「中國」、「企業」、「國際」、「發展」等，2010年有「民主」、「日本」，到了2015年出現了「鬼島」、「國防」、「國家」，最近的2020年則是「台灣人」、「普世」、「核心」等等。

![PTT 2005至2020年「台灣」一詞的鄰近詞變化](taiwan_nearest_neighbors.jpg)

```
# 下載 Gossiping 版 2005 至 2020 年，每五年的詞向量
!gdown --id "1gEL4v3wGgvqJnpWspISZvLeIL3GQZLB1" -O "Gossiping_2005.model" # 2005 年 Gossiping 板
!gdown --id "1yB9WPVDJVmmLLxbEHZroZP_cYMP0JUpC" -O "Gossiping_2010.model" # 2010 年 Gossiping 板
!gdown --id "1Vh8meq6hdte02nQ2-djclgpEKxFUC0YU" -O "Gossiping_2015.model" # 2015 年 Gossiping 板
!gdown --id "1EiDgWcnDDSOy1bu_aRjbBk4JGIENNoGk" -O "Gossiping_2020.model" # 2020 年 Gossiping 板

# 下載 WomenTalk 版 2005 至 2020 年，每五年的詞向量
!gdown --id "18rhI6VBnBXBji0YRplcL9bF31K2gFH9R" -O "WomenTalk_2005.model" # 2005 年 WomenTalk 板
!gdown --id "19XZ-SeZNUu515TZS3lW9kHASk_P6CYQJ" -O "WomenTalk_2010.model" # 2010 年 WomenTalk 板
!gdown --id "1CQtZ_5Tu8ML24es2vYfQcCoGcCTadzCp" -O "WomenTalk_2015.model" # 2015 年 WomenTalk 板
!gdown --id "1PqqW_5TyNKDU3WPubypIBED2GnlfFGTE" -O "WomenTalk_2020.model" # 2020 年 WomenTalk 板
```

```python
board_lst = ['Gossiping', 'WomenTalk']
year_lst = ['2005', '2010', '2015', '2020']
```

```python
import gensim # 讀入詞向量
```

```python
# 建立一個 class 來存放與詞向量有關的資料
class Embedding:
    def __init__(self, board, year_lst):
        self.board = board # 選定 PTT 的版，存成 string
        self.year_lst = year_lst # 選定各年份，存成 list
        
        self.path_lst = [f'{board}_{year}.model' for year in self.year_lst] # 該版各年份的詞向量檔案路徑
        self.model_lst = [gensim.models.Word2Vec.load(path) for path in self.path_lst] # 依詞向量檔案路徑，讀入檔案

# 建立 Gossiping 版，2005 及 2015 的詞向量 class
embed_2005_2015 = Embedding('Gossiping', ['2005', '2015'])
```

```python
# 看 embed_2005_2015 的 model_lst
embed_2005_2015.model_lst
```
[<gensim.models.word2vec.Word2Vec at 0x7f6bc03f5390>,
 <gensim.models.word2vec.Word2Vec at 0x7f6bc03f5630>]

```python
# 找出 model_lst[0] 中，'台灣' 的前35個鄰近詞
embed_2005.wv.most_similar('台灣', topn=5)
```
[('中國', 0.8369995355606079),
 ('美國', 0.7618862390518188),
 ('日本', 0.7539179921150208),
 ('發展', 0.7530442476272583),
 ('迪士尼', 0.7488158345222473)]

```python
# 找出 model_lst[0] 中，'台灣' 的詞向量
embed_2005['台灣']
```
array([ 8.07284042e-02, -1.51521474e-01,  2.04981357e-01,  7.02845901e-02,
        1.51984051e-01, -1.63245201e-01,  5.30136488e-02,  1.59432009e-01,
       -2.24587411e-01,  1.71152994e-01, ...

```python
# source: https://github.com/sismetanin/word2vec-tsne
def tsne_plot_similar_words(labels, embedding_clusters, word_clusters, n1):    
    plt.figure(figsize=(9, 9)) # 設定空白畫布

    colors = cm.Accent(np.linspace(0, 1, len(labels))) # 依 labels 數量設定不同的顏色
    # source: https://matplotlib.org/3.1.1/gallery/color/colormap_reference.html
    arrow_lst = []
    for label, embeddings, words, color in zip(labels, embedding_clusters, word_clusters, colors):
        x = embeddings[:, 0]
        y = embeddings[:, 1]
        arrow_lst.append((x[0], y[0])) # 第 0 個點是關鍵詞本身，抓出此點的 x, y，存入 arrow_lst 中
        
        # 畫點
        plt.scatter(x[:1], y[:1], c=color, alpha=1, label=label)
        for i, word in enumerate(words):
            # 關鍵詞本身
            if i == 0:
                a = 1 # 透明度
                size = 28 # 字體大小
            # 將近鄰詞分層，調整透明度與字體大小
            elif i >= 1 and i <= n1:
                a = 0.85
                size = 16
            else:
                a = 0.35
                size = 16

            # 標詞
            plt.annotate(word, alpha=a, xy=(x[i], y[i]), xytext=(1, 1),
                         textcoords='offset points', ha='right', va='bottom', size=size, c=color)
    
    for c, i in zip(colors, range(len(arrow_lst))):
        try:
            # 劃上箭頭方向
            plt.annotate('', xy=(arrow_lst[i+1][0], arrow_lst[i+1][1]), xytext=(arrow_lst[i][0], arrow_lst[i][1]),
                         arrowprops=dict(facecolor=c, edgecolor=c, width=5, shrink=0.01, alpha=0.5))
            # source: https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.annotate.html
        except:
            pass
        
    plt.legend(loc=4)
    plt.grid(True)
    plt.axis('off')
    plt.show()

class PlotTemporalData(Embedding): # 從 Embedding 這個 class 繼續擴增 function
    def __init__(self, board, year_lst):
        super().__init__(board, year_lst)
        # self.vocab_lst = [model.wv.vocab for model in self.model_lst] # 每個詞向量的 vocabulary

    # 抓出詞向量中的點   
    def create_datapoints(self, keyword, n1=10, n2=15): 
        error_log = {} # 紀錄錯誤訊息
        
        labels = [] # 詞_年份
        
        word_clusters = [] # 詞
        embedding_clusters = [] # 向量

        # 第一層 for loop: 各年份
        for year, model in zip(self.year_lst, self.model_lst): # 將 self.year_lst 和 self.model_lst 一一對應
            
            label = f'{keyword}({year})'
            
            try: # 若是有任何錯誤（Exception as e），以 try-except 紀錄錯誤訊息（e），並存至 error_log 這個 dictionary
                # 關鍵詞
                words = [label]
                embeddings = [model[keyword]]
                
                # 第二層 for loop: 某年份的鄰近詞
                # 鄰近詞（前 n1+n2 個鄰近詞）
                for similar_word, _ in model.wv.most_similar(keyword, topn=n1+n2):
                    words.append(similar_word)
                    embeddings.append(model[similar_word])
                embedding_clusters.append(embeddings)
                word_clusters.append(words)
                
                labels.append(label)
            except Exception as e:
                error_log[label] = e
                
        print(error_log)
        self.error_log = error_log
        
        self.keyword = keyword
        self.labels = labels
        
        self.n1 = n1
        self.n2 = n2
        
        self.embedding_clusters = embedding_clusters
        self.word_clusters = word_clusters
        
    # 將點經過 t-SNE 處理
    def tsne(self):
        embedding_clusters = np.array(self.embedding_clusters)
        n, m, k = embedding_clusters.shape
        tsne_model_en_2d = TSNE(perplexity=15, n_components=2, init='pca', n_iter=3500, random_state=32)
        embeddings_en_2d = np.array(tsne_model_en_2d.fit_transform(embedding_clusters.reshape(n * m, k))).reshape(n, m, 2)
        
        self.embeddings_en_2d = embeddings_en_2d
    
    # 將處理後的點視覺化
    def tsne_plot(self): 
        tsne_plot_similar_words(self.labels, self.embeddings_en_2d, self.word_clusters, self.n1)
```

```python
keyword = '台灣'
for board in board_lst:
    data = PlotTemporalData(board, year_lst)
    data.create_datapoints(keyword, n1=5, n2=5)
    #data.create_datapoints(keyword)
    data.tsne()
    data.tsne_plot()
```

除了以Word2Vec詞向量探討語意變遷外，亦可以BERT等並語境詞向量（contextualized word embeddings）將多義性（polysemy）的變動做形式表達\parencite{hu2019diachronic,giulianelli2019lexical}。

近年來的歷史詞彙語意研究，從詞意的改變、新舊字詞的興衰，探索其背後的運作機制與認知層面，已開始摸索出語意變遷（semantic change）的規律性（regularities）\parencite[63]{blank1999new}，例如：\textcite{dubossarsky2015bottom}、\textcite{hamilton2016law}、\textcite{xu2015computational}。
1.	\textcite{dubossarsky2015bottom}發現語意變遷的程度與字詞的原型（prototype）為正相關，越非典型的字詞，語意變遷的程度越高，為law of prototypicality。利用K-means分群分析找出字詞的分群，而分群的中心點代表的就是原型，也可以說是沒有成詞（non-lexicalized）的原型，因為這個中心點是分群分析數值計算下的一個點，可以想像該點是抽象的、不存在的一個字詞，而已成詞的原型是最靠近此中心點的另一個點。
2.	\textcite{hamilton2016law}發現，詞彙語意變遷的速度與使用詞頻呈現負相關，而在相同詞頻之下，越多詞意的字詞，語意變遷的程度越高，為law of conformity及law of innovation。從詞彙貢獻網路對應到詞彙的多義程度，再以迴歸分析找出語意變遷程度與詞頻與多義性的關聯。
3.	\textcite{dubossarsky2015bottom}則從近義詞（near-synonym）的角度切入，將law of parallel及law of differentiation兩相比較，發現law of parallel change比較明顯，也就是說近義詞之間的語意變化朝相似的語意發展。語意變遷的程度計算方法找出兩個時間點的共同鄰近詞比例為何，進而推論出近義詞之間隨時間變化的趨勢。

我們以歷時語料庫（中國哲學書電子計畫 \parencite{sturgeon2019ctext}）與現代漢語語料庫（中研院漢語平衡語料庫 \parencite{chen1996sinica}）為語料來源，建立歷時詞向量並搭配詞彙資料庫，並參考 \textcite{hamilton2016cultural} 的全域鄰近詞法，以搭配詞的相似度數值組成二階向量（second-order embedding），提高語意表徵的精確度來比較各時代向量的方法，求其相關係數和語意變遷程度之間的關聯。並從詞彙的意義分布與互動，描繪出不同詞意的消長與變動。

此外，採用以變異程度為基礎的近鄰群聚分析法（Variability-based Neighbor Clustering, VNC）\parencite{gries2012variability}，此階層式的分群可勾勒出綜合性評估各觀察變項的影響下，漢語詞彙發展的時代區分。

計算語意學與歷史語意學的整合研究可以使我們在經驗基礎上回溯驗證個別詞彙的意義變化，更進一步梳理整體的原理原則。詞彙反映人們對於新事物賦予新名的動機、社會概念的更迭也同時牽動詞彙之間的關聯，其應用範圍更可擴及到詞彙與文化變遷的探索。