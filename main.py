from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer
from PIL import Image
from io import BytesIO
import SocketServer
import base64
import numpy as np
from multiprocessing import Process, Queue

cnt = 0

class Handler(BaseHTTPRequestHandler):

    def __init__(self, request, client_address, server):
        BaseHTTPRequestHandler.__init__(self, request, client_address, server)

    def _set_headers(self):
        self.send_response(200)
        self.end_headers()

    def do_POST(self):
        global cnt, prev_image
        self._set_headers()
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        data = str(post_data)
        data = data[data.index(',')+1:]
        im = np.array(Image.open(BytesIO(base64.b64decode(data))))

        # for h in range(im.shape[0]):
            # for w in range(im.shape[1]):
                # if any(im[h,w,:]) != 0:
                    # print im[h,w,:]

        # Image.fromarray(im).save("figure_" +str(cnt) + ".png")
        # cnt += 1

address = ('localhost', 9090)
server = HTTPServer(address, Handler)
print "listening..."
server.serve_forever()
