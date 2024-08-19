// ==UserScript==
// @name         Keep Kaggle Notebook Alive
// @namespace    http://tampermonkey.net/
// @version      0.1
// @description  Keep the Kaggle Notebook alive by simulating user activity
// @author       liuweiqing
// @match        https://www.kaggle.com/*
// @grant        none
// ==/UserScript==

(function () {
  "use strict";

  // 每隔5分钟模拟一次点击事件，以保持Kaggle Notebook的活动状态
  setInterval(function () {
    console.log("Keeping the Kaggle Notebook alive...");
    const addButton = document.querySelector('button[aria-label="Add cell"]');
    if (addButton) {
      addButton.click(); // 模拟点击添加单元格按钮
    }
    const EnterEvent = new KeyboardEvent("keydown", {
      bubbles: true,
      cancelable: true,
      key: "Enter",
      code: "Enter",
      location: 0,
      ctrlKey: false,
      repeat: false,
    });
    const ctrlEnterEvent = new KeyboardEvent("keydown", {
      bubbles: true,
      cancelable: true,
      key: "Enter",
      code: "Enter",
      location: 0,
      ctrlKey: true, // 表示 Ctrl 键被按下
      repeat: false,
    });
    document.dispatchEvent(EnterEvent);
    setTimeout(() => {
      document.dispatchEvent(ctrlEnterEvent);
    }, 1000);
    const dKeyEvent = new KeyboardEvent("keydown", {
      key: "d",
      code: "KeyD",
      keyCode: 68, // 'D' 键的键码
      which: 68,
      bubbles: true,
      cancelable: true,
    });

    setTimeout(() => {
      targetElement.dispatchEvent(dKeyEvent);
      targetElement.dispatchEvent(dKeyEvent);
    }, 2000);
  }, 300000); // 300000 毫秒 = 5 分钟
})();
