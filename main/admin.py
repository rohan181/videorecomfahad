from django.contrib import admin

# Register your models here.




# Register your models here.
# 
# from import_export.admin import ImportExportModelAdmin

from .models import UserItem,Userprofile,Photo,Video



#@admin.register (Product)
#class ViewAdmin(ImportExportModelAdmin):
  #pass

#admin.site.register(Product,OrderAdmin)

admin.site.register(UserItem)
admin.site.register(Userprofile)
admin.site.register(Photo)
admin.site.register(Video)