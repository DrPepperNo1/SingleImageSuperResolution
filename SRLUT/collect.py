from lxml.html import etree
import requests
from PIL import Image
import os
import sys

path = './tutu'

if not os.path.exists(path):
    os.makedirs(path)

headers = {
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36'
}
url_list = []
page = 120
for area in ['4kfengjing', '4kyouxi', '4kdongman']:
        for i in range(0, page + 1):
            i = int(i)
            if i == 1:
                url = 'http://pic.netbian.com/' + area + '/index.html'
                res = requests.get(url=url, headers=headers)
                res.encoding = res.apparent_encoding  # 修改编码
                response = etree.HTML(res.text)

                response = etree.tostring(response)
                response = etree.fromstring(response)  # 以上搭建xpath对象
                content = response.xpath('//ul[@class="clearfix"]/li')
                for i in content:
                    tu_url = i.xpath('./a/@href')
                    tupian_url = 'http://pic.netbian.com' + ''.join(tu_url)
                    url_list.append(tupian_url)

            elif i >= 1:
                i = str(i)
                url = 'http://pic.netbian.com/' + area + '/index_' + i + '.html'
                res = requests.get(url=url, headers=headers)
                res.encoding = res.apparent_encoding  # 修改编码
                response = etree.HTML(res.text)
                response = etree.tostring(response)
                response = etree.fromstring(response)  # 以上搭建xpath对象
                content = response.xpath('//ul[@class="clearfix"]/li')
                for i in content:
                    tu_url = i.xpath('./a/@href')
                    tupian_url = 'http://pic.netbian.com' + ''.join(tu_url)
                    #url_list.append(tupian_url)

                    r = requests.get(url=tupian_url, headers=headers)
                    r.encoding = r.apparent_encoding  # 修改编码
                    html = etree.HTML(r.text)
                    html = etree.tostring(html)
                    html = etree.fromstring(html)  # 以上搭建xpath对象
                    url = html.xpath(r'//a[@id="img"]/img/@src')
                    rr = requests.get('http://pic.netbian.com' + ''.join(url), headers=headers)
                    name = html.xpath(r'//a[@id="img"]/img/@title')
                    rr.encoding = rr.apparent_encoding  # 修改编码

                    with open(f"./tutu/{''.join(name)}" + '.png', 'wb') as fw:
                        fw.write(rr.content)
                    img = Image.open(f"./tutu/{''.join(name)}" + '.png')
                    if img.size==(3840,2160):
                        img.save(f"./tutu/{''.join(name)}" + '.png', quality=100)
                        print(str(name) + " 保存完成！")
