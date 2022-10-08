function rand_max(v)
    idx = findall(v .== maximum(v))
    if length(idx) == 1
        idx[1]
    else
        rand(idx)
    end
end