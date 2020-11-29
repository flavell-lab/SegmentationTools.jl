struct WormOutOfFocus <: Exception
    idx::Int
    WormOutOfFocus(idx) = new(idx)
end

struct InsufficientCropping <: Exception
    idx::Int
    InsufficientCropping(idx) = new(idx)
end
