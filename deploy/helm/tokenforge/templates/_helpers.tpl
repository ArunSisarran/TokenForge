{{/* The engine Service's name: a different Service per mode (see engine.yaml). */}}
{{- define "tokenforge.engineService" -}}
{{ .Release.Name }}-engine{{ if eq .Values.engine.mode "external" }}-external{{ end }}
{{- end }}
