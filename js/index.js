"use strict";

document.body.classList.add("loading");

let resolution = {width: {exact: 640}, height: {exact: 480}};

let streaming = false;
let videoHeight, videoWidth;

let video = document.getElementById("video");
let canvasOutput = document.getElementById("canvasOutput");
let canvasOutputCtx = canvasOutput.getContext("2d");
let stream = null;

let info = document.getElementById("info");

function startCamera() {
    if (streaming) return;
    navigator.mediaDevices.getUserMedia({video: resolution, audio: false})
        .then(function(s) {
        stream = s;
        video.srcObject = s;
        video.play();
    })
        .catch(function(err) {
        console.log("An error occured! " + err);
    });
  
    video.addEventListener("canplay", function(ev){
        if (!streaming) {
            videoWidth = video.videoWidth;
            videoHeight = video.videoHeight;
            video.setAttribute("width", videoWidth);
            video.setAttribute("height", videoHeight);
            canvasOutput.width = videoWidth;
            canvasOutput.height = videoHeight;
            streaming = true;
        }
        startVideoProcessing();
    }, false);
}

let stats = null;

let faceClassifier = null;

let srcMat = null;
let grayMat = null;

let canvasInput = null;
let canvasInputCtx = null;

let canvasBuffer = null;
let canvasBufferCtx = null;

function startVideoProcessing() {
    if(!streaming) {console.warn("Please startup your webcam"); return;}
    stopVideoProcessing();
    canvasInput = document.createElement("canvas");
    canvasInput.width = videoWidth;
    canvasInput.height = videoHeight;
    canvasInputCtx = canvasInput.getContext("2d");

    canvasBuffer = document.createElement("canvas");
    canvasBuffer.width = videoWidth;
    canvasBuffer.height = videoHeight;
    canvasBufferCtx = canvasInput.getContext("2d");

    srcMat = new cv.Mat(videoHeight, videoWidth, cv.CV_8UC4);
    grayMat = new cv.Mat(videoHeight, videoWidth, cv.CV_8UC1);

    faceClassifier = new cv.CascadeClassifier();
    faceClassifier.load("cascade.xml");

    requestAnimationFrame(processVideo);
}

function processVideo() {
    stats.begin();
    
    canvasInputCtx.drawImage(video, 0, 0, videoWidth, videoHeight);
    let imageData = canvasInputCtx.getImageData(0, 0, videoWidth, videoHeight);
    srcMat.data.set(imageData.data);
    cv.cvtColor(srcMat, grayMat, cv.COLOR_RGBA2GRAY);
    
    let faces = [];
    
    let faceVect = new cv.RectVector();
    let faceMat = new cv.Mat();
    let size;

    cv.equalizeHist(grayMat, faceMat);
    //cv.pyrDown(grayMat, faceMat);
    //if(videoWidth > 320)
    //   cv.pyrDown(faceMat, faceMat);
    size = faceMat.size();
    let minsize = new cv.Size(size.height*.5, size.height*.5);

    faceClassifier.detectMultiScale(faceMat, faceVect, 1.1, 3, 0, minsize);
    for(let i = 0; i < faceVect.size(); i++) {
        let face = faceVect.get(i);
        faces.push(new cv.Rect(face.x, face.y, face.width, face.height));
    }
    faceMat.delete();
    faceVect.delete();

    canvasOutputCtx.drawImage(canvasInput, 0, 0, videoWidth, videoHeight);
    drawResults(canvasOutputCtx, faces, "lime", size);
    stats.end();
    requestAnimationFrame(processVideo);
}

function drawResults(ctx, results, color, size) {
    for(let i = 0; i < results.length; ++i) {
        let rect = results[i];
        let xRatio = videoWidth/size.width;
        let yRatio = videoHeight/size.height;
        ctx.lineWidth = 2;
        ctx.strokeStyle = color;
        ctx.strokeRect(rect.x*xRatio-rect.width*xRatio*.1, rect.y*yRatio-rect.height*yRatio*.1, rect.width*xRatio*1.2, rect.height*yRatio*1.2);
    }
}

function stopVideoProcessing() {
    if(srcMat != null && !srcMat.isDeleted()) srcMat.delete();
    if(grayMat != null && !grayMat.isDeleted()) grayMat.delete();
    if(faceClassifier != null && !faceClassifier.isDeleted()) faceClassifier.delete();
}

function initUI() {
    stats = new Stats();
    stats.showPanel(0);
    document.getElementById("mainView").appendChild(stats.dom);
}

function onOpenCvReady() {
    document.body.classList.remove("loading");
    if(!featuresReady) {
        info.innerHTML = "Required features are not ready.";
        return;
    }
    info.innerHTML = "Get ready to be judged!";
    initUI();
    startCamera();
}

