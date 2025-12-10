import { ChangeEvent, FC, useEffect, useState } from 'react';
import { SubmitHandler, useForm } from 'react-hook-form';
import { useGetGenModels } from '../../core/api';
import { GenerationModelApi, GenModel, SupportedAPI } from '../../types';

type FormValues = { model: string; name?: string; endpoint?: string; credentials?: string };

export const GenModelSetupForm: FC<{
  add: (model: Omit<GenModel & { api: SupportedAPI }, 'id'>) => void;
  cancel: () => void;
}> = ({ add, cancel }) => {
  const [availableAPIs, setAvailableAPIs] = useState<GenerationModelApi[]>([]);
  const [selectedAPI, setSelectedAPI] = useState<GenerationModelApi>(availableAPIs[0]);
  const [modelName, setModelName] = useState<string>('');
  const { models } = useGetGenModels();
  const { register, handleSubmit } = useForm<FormValues>();
  useEffect(() => {
    const fetchModels = async () => {
      setAvailableAPIs(await models());
    };
    fetchModels();
  }, [models]);
  useEffect(() => {
    setSelectedAPI(availableAPIs[0]);
  }, [availableAPIs]);

  const onAPIChange = (e: ChangeEvent<HTMLSelectElement>) => {
    const index = parseInt(e.target.value);
    if (index >= availableAPIs.length) {
      throw new Error(`Invalid index choice: ${index}`);
    }
    setSelectedAPI(availableAPIs[index]);
  };

  const onSubmit: SubmitHandler<FormValues> = (data: FormValues) => {
    const slug = data.model;
    const name = modelName;
    const endpoint = data.endpoint;
    const credentials = data.credentials;
    if (slug === null) throw new Error('You should provide a model');
    add({
      slug,
      name,
      api: selectedAPI.name as SupportedAPI,
      endpoint,
      credentials,
    });
  };

  const onNameChange = (e: ChangeEvent<HTMLInputElement>) => {
    setModelName(e.target.value);
  };

  const onModelChange = (e: ChangeEvent<HTMLSelectElement | HTMLInputElement>) => {
    setModelName(selectedAPI.name + '-' + e.target.value);
  };

  return (
    <form onSubmit={handleSubmit(onSubmit)}>
      <label htmlFor="api">API </label>
      <select id="api" defaultValue={0} onChange={onAPIChange} name="api">
        {availableAPIs.map((api, index) => (
          <option key={index} value={index}>
            {api.name}
          </option>
        ))}
      </select>
      {(() => {
        const inputs = [];
        if (selectedAPI !== undefined) {
          if (selectedAPI.name === 'OpenAI' || selectedAPI.name === 'ilaas')
            inputs.push(
              <div>
                <label htmlFor="model">Model</label>
                <select id="model" {...register('model', { onChange: onModelChange })}>
                  {selectedAPI.models.map((model) => (
                    <option key={model.slug} value={model.slug}>
                      {model.name}
                    </option>
                  ))}
                </select>
              </div>,
            );
          else {
            inputs.push(
              <div>
                <label htmlFor="model">Model</label>
                <input
                  type="text"
                  id="model"
                  placeholder="ID of the model"
                  {...register('model', { onChange: onModelChange })}
                />
              </div>,
            );
            if (selectedAPI.name !== 'OpenRouter')
              inputs.push(
                <div>
                  <label htmlFor="endpoint">Endpoint</label>
                  <input
                    type="text"
                    id="endpoint"
                    placeholder="enter the url of the endpoint"
                    {...register('endpoint')}
                  />
                </div>,
              );
          }

          if (selectedAPI.name !== 'Ollama')
            inputs.push(
              <div>
                <label htmlFor="credentials">API Credentials</label>
                <input
                  type="text"
                  id="credentials"
                  placeholder="API key"
                  autoComplete="off"
                  {...register('credentials')}
                />
              </div>,
            );
        }
        return inputs;
      })()}
      <label htmlFor="name">Name</label>
      <input
        type="text"
        id="name"
        value={modelName}
        {...register('name', { onChange: onNameChange })}
      />

      <button type="submit" className="btn-submit">
        Add
      </button>
    </form>
  );
};
