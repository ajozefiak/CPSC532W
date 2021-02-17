using JSON

function read_json(path)
    stringdata = join(readlines(path))
    dict = JSON.parse(stringdata)
    return dict
end
