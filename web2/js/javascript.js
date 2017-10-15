function gan() {
  var http = new XMLHttpRequest();
  //alert("test");
  var url = "http://localhost:5000";
  var table = $("#gantable");
  // http.open("POST", url, true);
  // http.setRequestHeader("Content-Type", "application/json");
  //
  // http.onload = function() {
  //   if (http.status == 200) {
  //     //table.find('tbody').append(http.responseText);
  //     alert("What the hell man");
  //     //document.getElementById("compilertext").value = temp.result;
  //   }
  //   else {
  //     alert('it failed');
  //     //table.find('tbody').append("HTTP Request failed. Check Docker server");
  //   }
  // };
  $.post(url, function(result) {
    result = JSON.parse(result);
    for (var i = 0; i < result[0].length; i++) {
      table.append("<td>" + result[0][i] + "</td>");
    }
    //table.find('tbody').append()
  });

  //var obj = { "code": myCode, "lang": lang};
  //http.send(JSON.stringify(obj));
}
/*
function handle(data) {
  table.clone('')
}*/
