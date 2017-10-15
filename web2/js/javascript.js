function gan() {
  var http = new XMLHttpRequest();
  var url = "http://localhost:5000";
  var table = $("#gantable");
  $.post(url, function(result) {
    result = JSON.parse(result);
    for (var i = 0; i < result[0].length; i++) {
      table.append("<td>" + result[0][i] + "</td>");
    }
  });
}
