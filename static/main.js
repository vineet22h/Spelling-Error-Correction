$(function convert() {

    console.log('inside');
    var form_data = new FormData();
    var ins = document.getElementById('ins').value;
    console.log(document.getElementById('ins').value);

    form_data.append('input', ins);

    $.ajax({
        url: '/convert',
        dataType: 'json',
        cache: false,
        contentType: false,
        processData: false,
        type: "POST",
        data: form_data,
        success: function(response) {
            console.log('insidsf');
            console.log(response);

            document.getElementById('out').value = response['output'];
        },
        error: function(response) {
            console.log('error in  convert');
            console.log(response);
        },
        complete: function() {
            setTimeout(convert, 1000);
        }
    });
});