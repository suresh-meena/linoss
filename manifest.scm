(use-modules (gnu packages python)
             (gnu packages rsync)
             (gnu packages ssh)
             (gnu packages version-control))

(packages->manifest
 (list git
       openssh
       python
       rsync))
