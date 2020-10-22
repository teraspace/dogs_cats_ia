puts gets

# Complete the function below.

# DO NOT MODIFY anything outside the below function
def twins(a, b)
    return 'No' if a.length != b.length
    return 'Yes' if a == b
    twin_result = 'No' #default
    a_arr = a.split('')
    b_arr = b.split('')
    a_arr.each_with_index do |a1, i|
      if a1 == b_arr[i]
        # do nothing
      else
        if i % 2 == 0
          # swap even
          a_arr.each_with_index do |a2, k|
            next if k % 2 != 0
            if b_arr[k] == a1
              a_arr[i] = a2
              a_arr[k] = a1
              if a == b
                twin_result = 'Yes'
                break # incomplete
              end
            end
          end
        else
          # swap odd
          a_arr.each_with_index do |a2, k|
            next if k % 2 == 0
            if b_arr[k] == a1
              a_arr[i] = a2
              a_arr[k] = a1
              if a == b
                twin_result = 'No'
                break # incomplete
              end
            end
          end
        end
      end
    end
    return twin_result
  end
# DO NOT MODIFY anything outside the above function

# DO NOT MODIFY THE CODE BELOW THIS LINE!
# _a_cnt = Integer(gets)
# _a_i=0
# _a = Array.new(_a_cnt)

# while (_a_i < _a_cnt)
#   _a_item = gets.to_s.strip;
#   _a[_a_i] = (_a_item)
#   _a_i+=1
# end

# _b_cnt = Integer(gets)
# _b_i=0
# _b = Array.new(_b_cnt)

# while (_b_i < _b_cnt)
#   _b_item = gets.to_s.strip;
#   _b[_b_i] = (_b_item)
#   _b_i+=1
# end
_a = ["cdab", "dcba"]
_b = ["abcd", "abcd"]
res = twins(_a, _b);
for res_i in res do
  puts res_i
end