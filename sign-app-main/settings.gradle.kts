pluginManagement {
    repositories {
        google {
            content {
                includeGroupByRegex("com\\.android.*")
                includeGroupByRegex("com\\.google.*")
                includeGroupByRegex("com\\.google.*")
                includeGroupByRegex("androidx.*")
            }
        }
        mavenCentral()
        gradlePluginPortal()
    }
}
dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()

        // [필살기] 이@게 없어서 uCrop을 못 사왔던 거다 이말이야! 똬악 박아라!
        maven { url = uri("https://jitpack.io") }
    }
}

rootProject.name = "sign-app"
include(":app")