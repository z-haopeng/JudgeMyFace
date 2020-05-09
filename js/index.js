"use strict";

document.body.classList.add("loading");

let streaming = false;
let videoHeight, videoWidth;

let video = document.getElementById("video");
let canvasOutput = document.getElementById("canvasOutput");
let canvasOutputCtx = canvasOutput.getContext("2d");
let stream = null;

let info = document.getElementById("info");
let scoreOutput = document.getElementById("scoreOutput");

let myBar = document.getElementById("myBar");

// Restarts the camera when window is resized (ie phone is rotated)
window.addEventListener("resize", function() {
    if(stopCamera()) {
        startCamera();
    }
});

function startCamera() {
    let qvga = {width: {exact: 320}, height: {exact: 240}};
    let vga = {width: {exact: 640}, height: {exact: 480}};
    let resolution = window.innerWidth < 640 ? qvga : vga;
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

let net = null;

let paused = false;

function startVideoProcessing() {
    if(!streaming) {console.warn("Please startup your webcam"); return;}
    stopVideoProcessing();
    canvasInput = document.createElement("canvas");
    canvasInput.width = videoWidth;
    canvasInput.height = videoHeight;
    canvasInputCtx = canvasInput.getContext("2d");

    net = cv.readNetFromCaffe("deploy.prototxt", "model.caffemodel");

    srcMat = new cv.Mat(videoHeight, videoWidth, cv.CV_8UC4);
    grayMat = new cv.Mat(videoHeight, videoWidth, cv.CV_8UC1);

    faceClassifier = new cv.CascadeClassifier();
    faceClassifier.load("cascade.xml");

    requestAnimationFrame(processVideo);
}

let threshold = 0;

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
    size = faceMat.size();
    
    // Speed hacks: only look for faces that are 50% the height of the picture, much less intensive
    let minsize = new cv.Size(size.height*.5, size.height*.5);
    faceClassifier.detectMultiScale(faceMat, faceVect, 1.1, 3, 0, minsize);
    
    for(let i = 0; i < faceVect.size(); i++) {
        let face = faceVect.get(i);
        faces.push(new cv.Rect(face.x, face.y, face.width, face.height));
    }
    faceMat.delete();
    faceVect.delete();

    if(!paused) {
        // Nerual net is very slow, so only run it once after a face has been on screen for a while
        if(threshold < 500) {
            if(faces.length > 0)
                threshold += 2;
            else if(threshold > 0)
                threshold--;
        } else if(faces.length > 0) {
            threshold = 0;
            rate(srcMat.roi(faces[0]));
            // Pause the video feed for 5 seconds
            paused = true;
            setTimeout(function() {
                paused = false;
                scoreOutput.innerHTML = "";
            }, 5000);
        }
        myBar.style.width = (threshold/5) + "%";
        canvasOutputCtx.drawImage(canvasInput, 0, 0, videoWidth, videoHeight);
        // Draw a red box in the picture that's actually analyzed
        if(paused)
            drawResults(canvasOutputCtx, faces, "red", size);
        else
            drawResults(canvasOutputCtx, faces, "lime", size);
    }
    stats.end();
    requestAnimationFrame(processVideo);
}

// Draws rectangles to the output
function drawResults(ctx, results, color, size) {
    for(let i = 0; i < results.length; ++i) {
        let rect = results[i];
        let xRatio = videoWidth/size.width;
        let yRatio = videoHeight/size.height;
        ctx.lineWidth = 3;
        ctx.strokeStyle = color;
        ctx.strokeRect(rect.x*xRatio, rect.y*yRatio, rect.width, rect.height);
    }
}

// Feeds the input Mat into the neural net
function rate(frame) {
    // Convert the color so it'll be acceptable input
    cv.cvtColor(frame, frame, cv.COLOR_RGBA2BGR);
    // Convert to YUV, equalize Y channel, and convert back to balance out brightness levels
    let yuvPlanes = new cv.MatVector();
    cv.cvtColor(frame, frame, cv.COLOR_BGR2YUV);
    cv.split(frame, yuvPlanes);
    let Y = yuvPlanes.get(0);
    cv.equalizeHist(Y, Y);
    cv.merge(yuvPlanes, frame);
    cv.cvtColor(frame, frame, cv.COLOR_YUV2BGR);
    // Cleanup
    yuvPlanes.delete();
    Y.delete();

    let blob = cv.blobFromImage(frame, 1, new cv.Size(224,224), [104, 117, 123, 0], false, false);
    net.setInput(blob);
    let out = net.forward();
    blob.delete();
    // It's technically an array, but there's always exactly one value in it
    // Also it outputs values up to 50, but I'm fitting it to a 5 point scale
    let score = Math.floor(out.data32F[0]*10)/100;
    scoreOutput.innerHTML = score.toFixed(2);
    out.delete();
}

// Cleanup OpenCV resources
function stopVideoProcessing() {
    if(srcMat != null && !srcMat.isDeleted()) srcMat.delete();
    if(grayMat != null && !grayMat.isDeleted()) grayMat.delete();
    if(faceClassifier != null && !faceClassifier.isDeleted()) faceClassifier.delete();
    if(net != null && !net.isDeleted()) net.delete();
}

function stopCamera() {
    if(streaming) {
        stopVideoProcessing();
        streaming = false;
        return true;
    }
    return false;
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
    info.innerHTML = "The Judgemental Webcam";
    initUI();
    startCamera();
}