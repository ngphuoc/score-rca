        Zygote.ignore() do
            function pow(x, n)
                r = one(x)
                for i = 1:n
                    r *= x
                end
                return r
            end
            gradient(5) do x
                pow(x, 2)
            end
        end


