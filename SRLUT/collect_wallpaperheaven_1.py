import requests      #爬虫
import re            #正则表达式
import os            #文件操作
from PIL import Image
import _thread
def get_url(base_url):
    #keyword=input("请输入英文关键词:(爬取排行榜请输入toplist)")
    keyword='random'
    if keyword=='random': 	#获取排行榜的url模板
        base_url=base_url+keyword+'?page='
    else: 					#获取基于关键词的url模板
        base_url=base_url+'search?q='+keyword+'&page='
    return base_url         #返回模板


def get_img_url(base_url,start_page=1,end_page=10):
    header={
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.182 Safari/537.36 Edg/88.0.705.74'
    } 									 #模拟浏览器头部，伪装成用户
    img_url_list=[]		 				 #创建一个空列表
    #for num in range(1,4): #循环遍历每页
    for num in range(start_page,end_page): #循环遍历每页
        new_url=base_url+str(num)  		 #将模板进行拼接得到每页壁纸完整的url(实质:字符串的拼接)
        page_text=requests.get(url=new_url,headers=header).text #获取url源代码
        ex='<a class="preview" href="(.*?)"'
        img_url_list+=re.findall(ex,page_text,re.S) 	#利用正则表达式从源代码中截取每张壁纸缩略图的url并全部存放在一个列表中
    return img_url_list					 #返回列表


def download_img(img_url_list):
    header={
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.182 Safari/537.36 Edg/88.0.705.74'
    } 											#模拟浏览器头部，伪装成用户
    keyword='hhh'
    if not os.path.exists('./downloadedimages'):     #在D盘目录下创建一个名为wallpapers的文件夹
        os.mkdir('./downloadedimages')
    path='./downloadedimages/'+keyword
    #if not os.path.exists(path):				#在wallpapers文件夹下创建一个以关键词命名的子文件夹以存放此次下载的所有壁纸
        #os.mkdir(path)
    for i in range(len(img_url_list)): 			#循环遍历列表，对每张壁纸缩略图的url进行字符串的增删获得壁纸原图下载的url  注：jpg或png结尾
        x=img_url_list[i].split('/')[-1]  		#获取最后一个斜杠后面的字符串
        a=x[0]+x[1] 							#获取字符串的前两位
        img_url='https://w.wallhaven.cc/full/'+a+'/wallhaven-'+x+'.png'  #拼接字符串,先默认jpg结尾
        code=requests.get(url=img_url,headers=header).status_code
        if code!=404:						    #若网页返回值为404，则为png结尾
            img_data=requests.get(url=img_url,headers=header,timeout=40).content  #获取壁纸图片的二进制数据,加入timeout限制请求时间
            img_name=img_url.split('-')[-1] 		#生成图片名字
            img_path=path+'/'+img_name			    #生成图片存储路径
            with open(img_path,'wb') as fp:		    #('w':写入,'b':二进制格式)
                fp.write(img_data)
            img = Image.open(img_path)
            if img.size != (3840, 2160):#删除尺寸不是4K的图
                del img#删除文件之前先关闭打开的图
                os.remove(img_path)


def main(url='https://wallhaven.cc/',page_min=600,page_max=10000):
    start_page = 0
    end_page = 2
    for i in range(page_min,page_max,2):
        print(i)
        base_url=get_url(url)
        img_url_list=get_img_url(base_url,start_page+i,end_page+i)
        download_img(img_url_list)

main('https://wallhaven.cc/')
'''
try:
   _thread.start_new_thread( main, ('https://wallhaven.cc/', 1,999))
   _thread.start_new_thread( main, ('https://wallhaven.cc/', 1001,1999))
   _thread.start_new_thread( main, ('https://wallhaven.cc/', 2001,2999))
   _thread.start_new_thread( main, ('https://wallhaven.cc/', 3001,3999))
   _thread.start_new_thread( main, ('https://wallhaven.cc/', 4001,4999))
   _thread.start_new_thread( main, ('https://wallhaven.cc/', 5001,5999))
   _thread.start_new_thread( main, ('https://wallhaven.cc/', 6001,6999))
   _thread.start_new_thread( main, ('https://wallhaven.cc/', 7001,7999))
   _thread.start_new_thread( main, ('https://wallhaven.cc/', 8001,8999))
   _thread.start_new_thread( main, ('https://wallhaven.cc/', 9001,9999))
   _thread.start_new_thread( main, ('https://wallhaven.cc/', 10001,10999))
   _thread.start_new_thread( main, ('https://wallhaven.cc/', 11001,11999))
   _thread.start_new_thread( main, ('https://wallhaven.cc/', 12001,12999))
   _thread.start_new_thread( main, ('https://wallhaven.cc/', 13001,13999))
except:
   print ("Error: unable to start thread")
 
while 1:
   pass
'''